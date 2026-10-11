import json
import math
import os
import re
import selectors
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 1024 * 1024


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DockerRunner:
    image: str
    network: str = "none"
    model: str = ""
    model_base_url: str = "http://model-gateway:8081/v1"
    assets_dir: Path | None = None
    timeout: float = 300
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
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.pids_limit <= 0:
            raise ValueError("invalid runner limits")
        if not re.fullmatch(r"[1-9][0-9]*[bBkKmMgG]?", self.memory):
            raise ValueError("invalid runner memory limit")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", self.cpus) or float(self.cpus) <= 0:
            raise ValueError("invalid runner cpu limit")

    def __call__(self, candidate_dir: Path, input_text: str) -> str:
        name = f"arena-runner-{uuid.uuid4().hex}"
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
        task = json.dumps(
            {
                "input": input_text,
                "model": self.model,
                "model_base_url": self.model_base_url,
            }
        ).encode()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError:
            raise RunnerError("runner unavailable") from None
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
            return answer
        except BaseException as error:
            self._stop(process, name)
            if isinstance(error, RunnerError):
                raise error from None
            if isinstance(error, Exception):
                raise RunnerError("runner failed") from None
            raise
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

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
