import json
import math
import os
import re
import selectors
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from evaluator.reliability import (
    MODEL_TIMEOUT,
    PREPARE_TIMEOUT,
    RETRY_DELAY,
    RUNNER_TIMEOUT,
    request_timeout,
)
from evaluator.usage import create_usage, read_usage, seal_usage

MAX_OUTPUT_BYTES = 1024 * 1024


class RunnerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, int] | None = None,
        wall_seconds: float | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.wall_seconds = wall_seconds
        self.run_id = run_id


@dataclass(frozen=True)
class SolveResult:
    answer: str
    usage: dict[str, int]
    wall_seconds: float


@dataclass(frozen=True)
class DockerRunner:
    image: str
    network: str = "none"
    model: str = ""
    model_base_url: str = "http://model-gateway:8081/v1"
    assets_dir: Path | None = None
    timeout: float = RUNNER_TIMEOUT
    model_timeout: float = MODEL_TIMEOUT
    retry_delay: float = RETRY_DELAY
    prepare_timeout: float = PREPARE_TIMEOUT
    usage_dir: Path | None = None
    memory: str = "2g"
    cpus: str = "2"
    pids_limit: int = 128

    def __post_init__(self) -> None:
        if not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[a-fA-F0-9]{64}", self.image
        ):
            raise ValueError("runner image must be pinned by sha256")
        if self.network == "host" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.network):
            raise ValueError("invalid runner network")
        if (
            any(
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
                for value in (self.timeout, self.model_timeout, self.prepare_timeout)
            )
            or isinstance(self.retry_delay, bool)
            or not math.isfinite(self.retry_delay)
            or self.retry_delay < 0
            or self.pids_limit <= 0
        ):
            raise ValueError("invalid runner limits")
        if not re.fullmatch(r"[1-9][0-9]*[bBkKmMgG]?", self.memory):
            raise ValueError("invalid runner memory limit")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", self.cpus) or float(self.cpus) <= 0:
            raise ValueError("invalid runner cpu limit")
        url = urlsplit(self.model_base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.path.rstrip("/") != "/v1"
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError("invalid runner model gateway URL")

    def __call__(self, candidate_dir: Path, input_text: str) -> SolveResult:
        started = time.monotonic()
        if self.usage_dir is None:
            raise RunnerError("runner usage directory is required")
        run_id = str(uuid.uuid4())
        name = f"arena-runner-{run_id}"
        command = [
            "docker",
            "run",
            "--name",
            name,
            "--rm",
            "--interactive",
            "--init",
            "--user",
            "10001:10001",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--network",
            self.network,
            "--log-driver",
            "none",
            "--workdir",
            "/tmp",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=128m,uid=10001,gid=10001",
            "--mount",
            self._mount(candidate_dir, "/candidate"),
        ]
        if self.assets_dir is not None:
            command.extend(["--mount", self._mount(self.assets_dir, "/assets")])
        command.append(self.image)
        url = urlsplit(self.model_base_url)
        scoped_url = urlunsplit(url._replace(path=f"/runs/{run_id}/v1"))
        task = json.dumps(
            {
                "input": input_text,
                "model": self.model,
                "model_base_url": scoped_url,
                "model_request_timeout": request_timeout(
                    self.model_timeout, retry_delay=self.retry_delay
                ),
                "prepare_timeout": self.prepare_timeout,
            }
        ).encode()
        try:
            create_usage(self.usage_dir, run_id)
        except (OSError, ValueError):
            raise RunnerError("runner usage unavailable") from None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError:
            with suppress(OSError, ValueError):
                seal_usage(self.usage_dir, run_id)
            raise self._failure(RunnerError("runner unavailable"), run_id, started) from None
        try:
            output = self._exchange(process, task)
            try:
                response = json.loads(output)
            except (ValueError, UnicodeError, RecursionError):
                raise RunnerError("invalid runner response") from None
            if not isinstance(response, dict):
                raise RunnerError("invalid runner response")
            answer = response.get("answer")
            if not isinstance(answer, str):
                raise RunnerError("invalid runner response")
            usage = seal_usage(self.usage_dir, run_id)
            return SolveResult(answer, usage, time.monotonic() - started)
        except BaseException as error:
            self._stop(process, name)
            with suppress(OSError, ValueError):
                seal_usage(self.usage_dir, run_id)
            if isinstance(error, RunnerError):
                raise self._failure(error, run_id, started) from None
            if isinstance(error, Exception):
                raise self._failure(RunnerError("runner failed"), run_id, started) from None
            raise
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def _failure(self, error: RunnerError, run_id: str, started: float) -> RunnerError:
        error.run_id = run_id
        error.wall_seconds = time.monotonic() - started
        if self.usage_dir is not None:
            with suppress(OSError, ValueError):
                error.usage = read_usage(self.usage_dir, run_id)
        return error

    @staticmethod
    def _mount(directory: Path, target: str) -> str:
        try:
            source = directory.resolve()
            valid = source.is_dir() and not any(
                character in str(source) for character in ",\n\r\x00"
            )
        except (OSError, RuntimeError):
            raise RunnerError("invalid runner mount") from None
        if not valid:
            raise RunnerError("invalid runner mount")
        return f"type=bind,src={source},dst={target},readonly"

    def _exchange(self, process: subprocess.Popen[bytes], task: bytes) -> bytes:
        assert process.stdin is not None
        assert process.stdout is not None
        deadline = time.monotonic() + self.timeout
        output = bytearray()
        sent = 0
        payload = memoryview(task)
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RunnerError("runner timed out")
                for key, _ in selector.select(remaining):
                    if key.fileobj is process.stdin:
                        try:
                            sent += os.write(key.fd, payload[sent : sent + 65536])
                        except BrokenPipeError:
                            sent = len(task)
                        if sent == len(task):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(process.stdout)
                        elif len(output) + len(chunk) > MAX_OUTPUT_BYTES:
                            raise RunnerError("runner output limit exceeded")
                        else:
                            output.extend(chunk)
        try:
            status = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise RunnerError("runner timed out") from None
        if status != 0:
            raise RunnerError("runner failed")
        return bytes(output)

    @staticmethod
    def _stop(process: subprocess.Popen[bytes], name: str) -> None:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            subprocess.run(
                ["docker", "rm", "--force", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
