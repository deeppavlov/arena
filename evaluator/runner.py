import hashlib
import json
import math
import os
import re
import selectors
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from evaluator.assets import AssetError, read_asset_manifest
from evaluator.build_receipts import bound_build_receipt, load_build_receipt
from evaluator.reliability import (
    MODEL_TIMEOUT,
    PREPARE_TIMEOUT,
    RETRY_DELAY,
    RUNNER_TIMEOUT,
    empty_usage,
    request_timeout,
)
from evaluator.usage import create_usage, read_usage, seal_usage

MAX_OUTPUT_BYTES = 1024 * 1024
MAX_TASK_BYTES = 8 * 1024 * 1024


def validate_runner_resources(
    memory: str, cpus: str, pids_limit: int, tmpfs_size: str, tmpfs_exec: bool,
) -> None:
    for name, value in (("memory", memory), ("tmpfs", tmpfs_size)):
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*[bBkKmMgG]?", value):
            raise ValueError(f"invalid runner {name} limit")
    if (
        not isinstance(cpus, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cpus)
        or not math.isfinite(float(cpus)) or float(cpus) <= 0
    ):
        raise ValueError("invalid runner cpu limit")
    if type(pids_limit) is not int or pids_limit <= 0 or type(tmpfs_exec) is not bool:
        raise ValueError("invalid runner resource limits")


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
    run_id: str | None = None


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
    uses_model: bool = True
    tmpfs_size: str = "128m"
    tmpfs_exec: bool = False
    published_assets_dir: Path | None = None
    build_dir: Path | None = None
    build_output_dir: Path | None = None
    supports_build: bool = False
    supports_asset_selection: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[a-fA-F0-9]{64}", self.image
        ):
            raise ValueError("runner image must be pinned by sha256")
        if self.network == "host" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.network):
            raise ValueError("invalid runner network")
        if type(self.supports_build) is not bool or type(self.supports_asset_selection) is not bool:
            raise ValueError("invalid runner capabilities")
        if self.supports_asset_selection and self.build_dir is None:
            raise ValueError("asset selection requires a trusted receipt ledger")
        if self.build_output_dir is not None and not self.supports_build:
            raise ValueError("runner does not support trusted builds")
        if type(self.uses_model) is not bool:
            raise ValueError("invalid runner model mode")
        if not self.uses_model and self.network != "none":
            raise ValueError("model-free runner requires network none")
        if (
            any(
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
                for value in (self.timeout, self.model_timeout, self.prepare_timeout)
            )
            or isinstance(self.retry_delay, bool)
            or not math.isfinite(self.retry_delay)
            or self.retry_delay < 0
        ):
            raise ValueError("invalid runner limits")
        validate_runner_resources(
            self.memory, self.cpus, self.pids_limit, self.tmpfs_size, self.tmpfs_exec,
        )
        url = urlsplit(self.model_base_url)
        if self.uses_model and (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.path.rstrip("/") != "/v1"
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError("invalid runner model gateway URL")
        if self.assets_dir is not None and self.published_assets_dir is not None:
            raw = self.assets_dir.resolve()
            published = self.published_assets_dir.resolve()
            if published.is_relative_to(raw) or raw.is_relative_to(published):
                raise ValueError("raw and published runner assets must be separate trees")

    def configuration(self) -> dict[str, Any]:
        return {
            name: str(value.resolve()) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }

    def __call__(
        self, candidate_dir: Path, input_text: str, *, parameters: dict[str, Any] | None = None,
    ) -> SolveResult:
        started = time.monotonic()
        if self.build_output_dir is not None and input_text:
            raise RunnerError("build input must be empty")
        selected_assets = self.validate_assets(parameters)
        prepared_assets = self.validate_preparation(candidate_dir, parameters)
        if self.uses_model and self.usage_dir is None:
            raise RunnerError("runner usage directory is required")
        run_id = str(uuid.uuid4())
        name = f"arena-runner-{run_id}"
        tmpfs_mode = "exec" if self.tmpfs_exec else "noexec"
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
            f"/tmp:rw,{tmpfs_mode},nosuid,nodev,size={self.tmpfs_size},uid=10001,gid=10001",
            "--mount",
            self._mount(candidate_dir, "/candidate"),
        ]
        if self.assets_dir is not None:
            command.extend(["--mount", self._mount(self.assets_dir, "/assets")])
        if selected_assets:
            command.extend([
                "--tmpfs", "/assets/derived:rw,noexec,nosuid,nodev,size=1m,mode=0555,uid=0,gid=0",
            ])
        for digest, directory in selected_assets:
            command.extend(["--mount", self._mount(directory, f"/assets/derived/{digest}")])
        if prepared_assets is not None:
            command.extend(["--mount", self._mount(prepared_assets, "/assets/prepared")])
        if self.build_output_dir is not None:
            command.extend([
                "--mount", self._mount(self.build_output_dir, "/output", readonly=False),
            ])
        command.append(self.image)
        payload: dict[str, Any] = {
            "input": input_text,
            "prepare_timeout": self.prepare_timeout,
        }
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise RunnerError("invalid harness parameters")
            payload["parameters"] = parameters
        if self.build_output_dir is not None:
            payload.pop("input")
            payload["operation"] = "build"
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            raise RunnerError("invalid harness parameters") from None
        if self.uses_model:
            model_request_timeout = request_timeout(
                self.model_timeout, retry_delay=self.retry_delay
            )
            assert self.usage_dir is not None
            try:
                capability = create_usage(self.usage_dir, run_id)
            except (OSError, ValueError):
                raise RunnerError("runner usage unavailable") from None
            url = urlsplit(self.model_base_url)
            scoped_url = urlunsplit(url._replace(path=f"/runs/{run_id}/{capability}/v1"))
            payload.update({
                "model": self.model,
                "model_base_url": scoped_url,
                "model_request_timeout": model_request_timeout,
            })
        else:
            payload["uses_model"] = False
        try:
            task = json.dumps(
                payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"),
            ).encode(errors="backslashreplace")
        except (TypeError, ValueError, RecursionError):
            with suppress(OSError, ValueError):
                self._seal(run_id)
            raise RunnerError("invalid harness parameters") from None
        if len(task) > MAX_TASK_BYTES:
            with suppress(OSError, ValueError):
                self._seal(run_id)
            raise self._failure(RunnerError("runner task too large"), run_id, started)
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
                self._seal(run_id)
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
            usage = self._seal(run_id)
            return SolveResult(answer, usage, time.monotonic() - started, run_id)
        except BaseException as error:
            self._stop(process, name)
            with suppress(OSError, ValueError):
                self._seal(run_id)
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

    def validate_assets(self, parameters: dict[str, Any] | None) -> list[tuple[str, Path]]:
        if parameters is not None and not isinstance(parameters, dict):
            raise RunnerError("invalid harness parameters")
        if not self.supports_asset_selection:
            return []
        digests = parameters.get("asset_digests", []) if parameters is not None else []
        if (
            not isinstance(digests, list) or len(digests) > 16
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
                   for value in digests)
            or digests != sorted(set(digests))
        ):
            raise RunnerError("invalid runner asset selection")
        if not digests:
            return []
        if self.assets_dir is not None:
            mountpoint = self.assets_dir / "derived"
            if mountpoint.is_symlink() or not mountpoint.is_dir():
                raise RunnerError("invalid derived runner asset mountpoint")
        return [(digest, self._published_asset(digest)) for digest in digests]

    def validate_preparation(
        self, candidate_dir: Path, parameters: dict[str, Any] | None,
    ) -> Path | None:
        if self.build_output_dir is not None:
            return None
        try:
            receipt = self.preparation_receipt(candidate_dir)
            if receipt is None and self.supports_asset_selection:
                for digest in (parameters or {}).get("asset_digests", []):
                    if bound_build_receipt(self.build_dir, digest) is not None:
                        if not self.supports_build:
                            raise RunnerError("imported preparation requires trusted build support")
                        raise RunnerError("imported preparation requires build.py")
        except (OSError, ValueError):
            raise RunnerError("candidate preparation unavailable") from None
        if receipt is None:
            return None
        if self.assets_dir is not None:
            mountpoint = self.assets_dir / "prepared"
            if mountpoint.is_symlink() or not mountpoint.is_dir():
                raise RunnerError("invalid prepared runner asset mountpoint")
        return self._published_asset(receipt["output_digest"])

    def preparation_receipt(
        self, candidate_dir: Path, candidate_digest: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.supports_build:
            return None
        return load_build_receipt(self.build_dir, candidate_dir, candidate_digest)

    def _published_asset(self, digest: str) -> Path:
        store = self.published_assets_dir
        if store is None or store.is_symlink() or not store.is_dir():
            raise RunnerError("published runner assets unavailable")
        directory = store / digest
        try:
            read_asset_manifest(directory, digest)
        except (AssetError, OSError, ValueError):
            raise RunnerError(f"published runner asset unavailable: {digest}") from None
        return directory

    def _seal(self, run_id: str) -> dict[str, int]:
        if not self.uses_model:
            return empty_usage()
        assert self.usage_dir is not None
        return seal_usage(self.usage_dir, run_id)

    def _failure(self, error: RunnerError, run_id: str, started: float) -> RunnerError:
        error.run_id = run_id
        error.wall_seconds = time.monotonic() - started
        if not self.uses_model:
            error.usage = empty_usage()
        elif self.usage_dir is not None:
            with suppress(OSError, ValueError):
                error.usage = read_usage(self.usage_dir, run_id)
        return error

    @staticmethod
    def _mount(directory: Path, target: str, *, readonly: bool = True) -> str:
        try:
            source = directory.resolve()
            valid = source.is_dir() and not any(
                character in str(source) for character in ",\n\r\x00"
            )
        except (OSError, RuntimeError):
            raise RunnerError("invalid runner mount") from None
        if not valid:
            raise RunnerError("invalid runner mount")
        return f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")

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


class HarnessRunner:
    def __init__(self, default_runner: DockerRunner, registry: dict[str, DockerRunner]) -> None:
        if any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or not isinstance(runner, DockerRunner)
            for name, runner in registry.items()
        ):
            raise ValueError("invalid harness registry")
        self.default_runner = default_runner
        self.registry = MappingProxyType(dict(registry))

    def provenance(self) -> dict[str, Any]:
        configuration = {
            "default": self.default_runner.configuration(),
            "registry": {name: runner.configuration() for name, runner in self.registry.items()},
        }
        digest = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return {**configuration, "digest": digest}

    def __call__(self, candidate_dir: Path, input_text: str) -> SolveResult:
        runner, parameters = self.selection(candidate_dir)
        return runner(candidate_dir, input_text, parameters=parameters)

    def validate_candidate(self, candidate_dir: Path) -> None:
        runner, parameters = self.selection(candidate_dir)
        runner.validate_assets(parameters)
        runner.validate_preparation(candidate_dir, parameters)

    def preparation_receipt(
        self, candidate_dir: Path, candidate_digest: str | None = None,
    ) -> dict[str, Any] | None:
        runner, _ = self.selection(candidate_dir)
        return runner.preparation_receipt(candidate_dir, candidate_digest)

    def build_asset_digests(self, candidate_dir: Path) -> list[str]:
        runner, parameters = self.selection(candidate_dir)
        if not runner.supports_build:
            raise RunnerError("selected harness does not support trusted builds")
        return [digest for digest, _ in runner.validate_assets(parameters)]

    def selection(self, candidate_dir: Path) -> tuple[DockerRunner, dict[str, Any] | None]:
        manifest = candidate_dir / "solution.json"
        if not manifest.exists():
            return self.default_runner, None
        try:
            with manifest.open("rb") as source:
                raw = source.read(MAX_OUTPUT_BYTES + 1)
            if len(raw) > MAX_OUTPUT_BYTES:
                raise ValueError
            selection = json.loads(raw)
            if not isinstance(selection, dict) or set(selection) - {"harness", "parameters"}:
                raise ValueError
            harness = selection.get("harness")
            if not isinstance(harness, str) or harness not in self.registry:
                raise ValueError
            parameters = selection.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError
            json.dumps(parameters, allow_nan=False)
        except (OSError, TypeError, ValueError, RecursionError):
            raise RunnerError("invalid harness selection") from None
        return self.registry[harness], parameters
