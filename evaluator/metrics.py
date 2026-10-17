import hashlib
import json
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import Field

from evaluator.contracts import CommandMetric, Example, Metric, StrictModel
from evaluator.judge import ModelClient, judge_answer
from evaluator.reliability import MODEL_RETRIES, RETRY_DELAY


class CommandScore(StrictModel):
    score: float = Field(strict=True, allow_inf_nan=False)
    statistics: dict[str, Any] = Field(default_factory=dict)


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


def _run_command(command: CommandMetric, payload: dict[str, Any]) -> dict[str, Any]:
    _command_provenance(command)
    result = subprocess.run(
        command.argv,
        input=json.dumps(payload, allow_nan=False),
        text=True,
        capture_output=True,
        timeout=command.timeout_seconds,
        check=True,
        cwd="/",
    )
    score = CommandScore.model_validate_json(result.stdout)
    json.dumps(score.statistics, allow_nan=False)
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
