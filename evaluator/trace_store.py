import json
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


def read_calls(directory: Path, run_id: str) -> list[dict[str, Any]]:
    calls = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(_directory(directory, run_id).glob("*.json"))
    ]
    return sorted(calls, key=lambda call: call.get("start_time_ns", 0))
