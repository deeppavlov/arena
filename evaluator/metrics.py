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
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from evaluator.contracts import CommandMetric, Example, Metric, StrictModel
from evaluator.judge import ModelClient, judge_answer
from evaluator.numeric import validate_score
from evaluator.reliability import MODEL_RETRIES, RETRY_DELAY

MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
MAX_COMMAND_DIAGNOSTIC_BYTES = 64 * 1024


class CommandMetricError(ValueError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class CommandScore(StrictModel):
    score: float = Field(strict=True, allow_inf_nan=False)
    statistics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def supported_score(cls, value: object) -> float:
        return validate_score(value)


def _command_provenance(command: CommandMetric) -> dict[str, Any]:
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


def _run_command(command: CommandMetric, payload: dict[str, Any]) -> dict[str, Any]:
    _command_provenance(command)
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
    diagnostics = {
        "command": command.argv,
        "returncode": process.returncode if process is not None else None,
        "timed_out": isinstance(execution_error, subprocess.TimeoutExpired),
        "timeout_seconds": command.timeout_seconds,
        "stdout": output[:MAX_COMMAND_DIAGNOSTIC_BYTES].decode(errors="replace"),
        "stderr": errors[:MAX_COMMAND_DIAGNOSTIC_BYTES].decode(errors="replace"),
        "stdout_truncated": len(output) > MAX_COMMAND_DIAGNOSTIC_BYTES,
        "stderr_truncated": len(errors) > MAX_COMMAND_DIAGNOSTIC_BYTES,
    }
    if execution_error is not None:
        raise CommandMetricError(str(execution_error), diagnostics) from execution_error
    if process is not None and process.returncode:
        raise CommandMetricError(
            f"metric command exited with status {process.returncode}", diagnostics,
        )
    if len(output) > MAX_COMMAND_OUTPUT_BYTES:
        raise CommandMetricError("metric command output exceeds 1 MiB", diagnostics)
    try:
        score = CommandScore.model_validate_json(output)
        json.dumps(score.statistics, allow_nan=False)
    except ValueError as error:
        raise CommandMetricError(str(error), diagnostics) from error
    return score.model_dump(exclude_unset=True)


def evaluate_answer(
    metric: Metric,
    example: Example,
    answer: str,
    complete: ModelClient | Callable[[list[dict[str, str]]], str] | None = None,
    *,
    retries: int = MODEL_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> dict[str, Any]:
    if metric.kind == "command":
        if metric.command is None:
            raise ValueError("command metric requires a command")
        result = _run_command(metric.command, {"example": example.model_dump(), "answer": answer})
        return {**result, "criteria": {}}
    if example.reference is None or not example.reference.strip():
        raise ValueError("LLM metric requires a nonblank reference")
    if not answer.strip():
        return {"score": 0.0, "criteria": {}, "failure": "candidate_failure"}
    if complete is None:
        raise ValueError("LLM metric requires a model")
    return judge_answer(metric, example, answer, complete, retries=retries, retry_delay=retry_delay)


def aggregate_results(metric: Metric, results: list[dict[str, Any]]) -> float:
    if not results:
        raise ValueError("results must not be empty")
    scores = [CommandScore.model_validate({"score": result["score"]}).score for result in results]
    if metric.aggregation is None:
        return statistics.mean(scores)
    return _run_command(metric.aggregation, {"results": results})["score"]
