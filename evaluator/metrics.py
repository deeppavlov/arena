import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from pydantic import Field, field_validator

from evaluator.contracts import CommandMetric, Example, Metric, StrictModel
from evaluator.judge import ModelClient, judge_answer
from evaluator.numeric import validate_score
from evaluator.reliability import (
    MODEL_RETRIES,
    MODEL_TIMEOUT,
    RETRY_DELAY,
    request_timeout,
    validate_timing,
)
from evaluator.usage import create_usage, read_usage, seal_usage

MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
MAX_COMMAND_DIAGNOSTIC_BYTES = 64 * 1024


@dataclass(frozen=True)
class MetricModelAccess:
    model: str
    base_url: str
    usage_dir: Path
    model_timeout: float = MODEL_TIMEOUT
    retry_delay: float = RETRY_DELAY

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if (
            not self.model.strip()
            or url.scheme not in {"http", "https"}
            or not url.hostname
            or url.port == 0
            or url.path.rstrip("/") != "/v1"
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValueError("invalid metric model access")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        object.__setattr__(
            self, "model_timeout", validate_timing(self.model_timeout, "model_timeout"),
        )
        object.__setattr__(
            self, "retry_delay", validate_timing(self.retry_delay, "retry_delay", allow_zero=True),
        )


class CommandMetricError(ValueError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class CommandStatistics(StrictModel):
    statistics: dict[str, Any]


class CommandScore(StrictModel):
    score: float = Field(strict=True, allow_inf_nan=False)
    statistics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def supported_score(cls, value: object) -> float:
        return validate_score(value)


def _command_provenance(command: CommandMetric) -> dict[str, Any]:
    for name in command.source_sha256:
        path = Path(name)
        if path.name == "metric.lock.json":
            from evaluator.metric_package import metric_package_provenance

            metric_package_provenance(path.parent)
    sources = {Path(name).resolve() for name in command.source_sha256}
    for argument in command.argv:
        path = Path("/") / argument
        if path.is_file() and path.resolve() not in sources:
            raise ValueError("command file arguments must be pinned in source_sha256")
    for name, expected in command.source_sha256.items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("command source hash differs from frozen configuration")
    return command.model_dump()


def command_provenance(metric: Metric) -> dict[str, Any] | None:
    if metric.command is None and metric.aggregation is None:
        return None
    return {
        name: _command_provenance(command) if command is not None else None
        for name, command in (("command", metric.command), ("aggregation", metric.aggregation))
    }


def _group_running(group_id: int) -> bool:
    for status in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = status.read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == group_id and fields[0] not in {b"Z", b"X"}:
            return True
    return False


def _stop_command(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()
    if sys.platform == "linux":
        while _group_running(process.pid):
            time.sleep(0.01)


def _redact_output(value: bytes, capability: str) -> bytes:
    try:
        value = json.dumps(json.loads(value)).encode()
    except (ValueError, RecursionError):
        pass
    escaped = "".join(f"\\u{ord(char):04x}" for char in capability).encode()
    return value.replace(capability.encode(), b"[redacted]").replace(escaped, b"[redacted]")


def _execute_command(
    command: CommandMetric,
    payload: dict[str, Any],
    *,
    allow_statistics: bool,
    capability: str | None,
) -> dict[str, Any]:
    encoded = json.dumps(payload, allow_nan=False).encode()
    process = None
    execution_error = None
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write(encoded)
        stdin.seek(0)
        try:
            process = subprocess.Popen(
                command.argv, stdin=stdin, stdout=stdout, stderr=stderr,
                start_new_session=True, cwd="/",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            try:
                process.wait(timeout=command.timeout_seconds)
            finally:
                _stop_command(process)
        except Exception as error:
            execution_error = error
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read(MAX_COMMAND_OUTPUT_BYTES + 1)
        errors = stderr.read(MAX_COMMAND_DIAGNOSTIC_BYTES + 1)
    output_size = len(output)
    error_size = len(errors)
    if capability is not None:
        output = _redact_output(output, capability)
        errors = _redact_output(errors, capability)
    diagnostics = {
        "command": command.argv,
        "returncode": process.returncode if process is not None else None,
        "timed_out": isinstance(execution_error, subprocess.TimeoutExpired),
        "timeout_seconds": command.timeout_seconds,
        "stdout": output[:MAX_COMMAND_DIAGNOSTIC_BYTES].decode(errors="replace"),
        "stderr": errors[:MAX_COMMAND_DIAGNOSTIC_BYTES].decode(errors="replace"),
        "stdout_truncated": output_size > MAX_COMMAND_DIAGNOSTIC_BYTES,
        "stderr_truncated": error_size > MAX_COMMAND_DIAGNOSTIC_BYTES,
    }
    if execution_error is not None:
        raise CommandMetricError(str(execution_error), diagnostics) from execution_error
    if process is not None and process.returncode:
        raise CommandMetricError(
            f"metric command exited with status {process.returncode}", diagnostics,
        )
    if output_size > MAX_COMMAND_OUTPUT_BYTES:
        raise CommandMetricError("metric command output exceeds 1 MiB", diagnostics)
    try:
        decoded = json.loads(output)
        if allow_statistics and isinstance(decoded, dict) and "score" not in decoded:
            score = CommandStatistics.model_validate(decoded)
        else:
            score = CommandScore.model_validate(decoded)
        json.dumps(score.statistics, allow_nan=False)
    except (ValueError, RecursionError) as error:
        raise CommandMetricError(str(error), diagnostics) from error
    return score.model_dump(exclude_unset=True)


def _run_command(
    command: CommandMetric,
    payload: dict[str, Any],
    *,
    allow_statistics: bool = False,
    model_access: MetricModelAccess | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    _command_provenance(command)
    if not command.uses_model:
        return _execute_command(
            command, payload, allow_statistics=allow_statistics, capability=None,
        )
    if model_access is None:
        raise ValueError("metric command requires scoped model access")
    timeout = request_timeout(model_access.model_timeout, retry_delay=model_access.retry_delay)
    run_id = str(uuid4())
    try:
        capability = create_usage(model_access.usage_dir, run_id)
    except (OSError, ValueError):
        raise CommandMetricError("metric model accounting unavailable", {
            "model_run_id": run_id, "model_usage_complete": False,
        }) from None
    scoped_url = urlunsplit(urlsplit(model_access.base_url)._replace(
        path=f"/runs/{run_id}/{capability}/v1",
    ))
    payload = {
        **payload,
        "model": model_access.model,
        "model_base_url": scoped_url,
        "model_request_timeout": timeout,
    }
    failure: BaseException | None = None
    try:
        return _execute_command(
            command, payload, allow_statistics=allow_statistics, capability=capability,
        )
    except CommandMetricError as error:
        failure = error
        raise
    except Exception as error:
        failure = CommandMetricError(str(error).replace(capability, "[redacted]"), {})
        raise failure from None
    except BaseException as error:
        failure = error
        raise
    finally:
        complete = True
        counters = None
        try:
            counters = seal_usage(model_access.usage_dir, run_id)
        except (OSError, ValueError):
            complete = False
            try:
                counters = read_usage(model_access.usage_dir, run_id)
            except (OSError, ValueError):
                pass
        if usage is not None and counters is not None:
            for key, value in counters.items():
                usage[key] = usage.get(key, 0) + value
        diagnostics = {"model_run_id": run_id, "model_usage_complete": complete}
        if isinstance(failure, CommandMetricError):
            failure.diagnostics.update(diagnostics)
        if not complete and failure is None:
            raise CommandMetricError("metric model accounting incomplete", diagnostics)


def evaluate_answer(
    metric: Metric,
    example: Example,
    answer: str,
    complete: ModelClient | Callable[[list[dict[str, str]]], str] | None = None,
    *,
    retries: int = MODEL_RETRIES,
    retry_delay: float = RETRY_DELAY,
    model_access: MetricModelAccess | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    if metric.kind == "command":
        if metric.command is None:
            raise ValueError("command metric requires a command")
        result = _run_command(
            metric.command, {"example": example.model_dump(), "answer": answer},
            allow_statistics=metric.aggregation is not None, model_access=model_access, usage=usage,
        )
        return {**result, "criteria": {}}
    if example.reference is None or not example.reference.strip():
        raise ValueError("LLM metric requires a nonblank reference")
    if not answer.strip():
        return {"score": 0.0, "criteria": {}, "failure": "candidate_failure"}
    if complete is None:
        raise ValueError("LLM metric requires a model")
    return judge_answer(metric, example, answer, complete, retries=retries, retry_delay=retry_delay)


def aggregate_results(
    metric: Metric,
    results: list[dict[str, Any]],
    *,
    model_access: MetricModelAccess | None = None,
    usage: dict[str, int] | None = None,
) -> float:
    if not results:
        raise ValueError("results must not be empty")
    if metric.aggregation is None:
        return statistics.mean(
            CommandScore.model_validate({"score": result["score"]}).score for result in results
        )
    return _run_command(
        metric.aggregation, {"results": results}, model_access=model_access, usage=usage,
    )["score"]
