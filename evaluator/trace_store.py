import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from evaluator.tracking import _save, redact


def _directory(directory: Path, run_id: str) -> Path:
    if str(UUID(run_id)) != run_id:
        raise ValueError("Trace run ID must be a canonical UUID")
    return directory / run_id


def append_call(
    directory: Path, run_id: str, call: dict[str, Any], *, secrets: tuple[str, ...] = (),
) -> None:
    target = _directory(directory, run_id) / f"{uuid4()}.json"
    _save(target, redact(call, secrets=secrets))


def call_paths(directory: Path, run_id: str) -> list[Path]:
    return sorted(_directory(directory, run_id).glob("*.json"))


def iter_calls(directory: Path, run_id: str) -> Iterator[dict[str, Any]]:
    for path in call_paths(directory, run_id):
        yield json.loads(path.read_text(encoding="utf-8"))


def read_calls(directory: Path, run_id: str) -> list[dict[str, Any]]:
    return sorted(iter_calls(directory, run_id), key=lambda call: call.get("start_time_ns", 0))
