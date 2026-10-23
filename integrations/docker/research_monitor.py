import argparse
import fcntl
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from evaluator.reliability import empty_usage
from evaluator.tracking import _save, export_record, flush_queue, sanitize_record
from evaluator.usage import _load


@dataclass(frozen=True)
class Settings:
    active_file: Path
    usage_dir: Path
    trace_dir: Path
    state_dir: Path
    params: dict[str, Any]
    shared_queues: tuple[Path, ...] = ()
    operator_queues: tuple[Path, ...] = ()

    @classmethod
    def load(cls, path: Path) -> Self:
        value = json.loads(path.read_text(encoding="utf-8"))
        params = value["params"]
        if not all(isinstance(params.get(key), str) and params[key]
                   for key in ("case_id", "campaign_id")):
            raise ValueError("Monitor requires explicit case and campaign IDs")
        return cls(
            active_file=Path(value["active_file"]), usage_dir=Path(value["usage_dir"]),
            trace_dir=Path(value["trace_dir"]), state_dir=Path(value["state_dir"]), params=params,
            shared_queues=tuple(Path(item) for item in value.get("shared_queues", [])),
            operator_queues=tuple(Path(item) for item in value.get("operator_queues", [])),
        )


def _inspect_container(container_id: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", container_id],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError("Research container inspection failed")
    state = json.loads(result.stdout)
    if not isinstance(state, dict) or "Status" not in state:
        raise ValueError("Invalid container state")
    return state


def _seal_when_idle(directory: Path, run_id: str) -> dict[str, int] | None:
    path = directory / f"{run_id}.json"
    with (directory / f"{run_id}.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = _load(path)
        if value["active_requests"]:
            return None
        if not value["closed"]:
            value["closed"] = True
            _save(path, value)
        return {key: int(value[key]) for key in empty_usage()}


def _timestamp_ns(value: str) -> int:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.year < 1970:
        raise ValueError("Invalid container timestamp")
    fractional = re.search(r"\.(\d+)", value)
    nanos = int((fractional[1] + "000000000")[:9]) if fractional else 0
    return int(stamp.replace(microsecond=0).timestamp()) * 1_000_000_000 + nanos


def _record(
    config: Settings, run_id: str, container_id: str, state: dict[str, Any], usage: dict[str, int],
) -> dict[str, Any]:
    call_files = sorted((config.trace_dir / run_id).resolve().glob("*.json"))
    started = _timestamp_ns(state["StartedAt"])
    finished = _timestamp_ns(state["FinishedAt"])
    for path in call_files:
        call = json.loads(path.read_text(encoding="utf-8"))
        started = min(started, call["start_time_ns"])
        finished = max(finished, call["end_time_ns"])
    if finished < started:
        raise ValueError("Container completion precedes start")
    wall_seconds = (finished - started) / 1_000_000_000
    success = state["Status"] == "exited" and state["ExitCode"] == 0 and not state["OOMKilled"]
    return sanitize_record({
        "split": "train", "status": "FINISHED" if success else "FAILED",
        "params": {**config.params, "candidate_id": "orchestrator", "role": "orchestrator",
                   "model_run_id": run_id},
        "metrics": {
            "orchestrator_tokens": usage["total_tokens"], "orchestrator_seconds": wall_seconds,
            "orchestrator_attempts": usage["attempts"],
            "orchestrator_trace_attempts": len(call_files),
            "orchestrator_unreported_attempts": usage["unreported_attempts"],
            "wall_seconds": wall_seconds,
        },
        "resources": {"orchestrator": {
            "run_id": run_id, "usage": usage, "wall_seconds": wall_seconds,
        }},
        "trace_complete": len(call_files) == usage["attempts"],
        "container_id": container_id, "exit_code": state["ExitCode"],
        "oom_killed": state["OOMKilled"], "calls": [],
        "call_files": [str(path) for path in call_files],
        "start_time_ns": started, "end_time_ns": finished,
    })


def _monitor(config: Settings) -> dict[str, Any]:
    if not config.active_file.exists():
        return {"status": "not_started"}
    active = json.loads(config.active_file.read_text(encoding="utf-8"))
    run_id = active["run_id"]
    if not isinstance(run_id, str) or str(UUID(run_id)) != run_id:
        raise ValueError("Monitor run ID must be a canonical UUID")
    container_id = active.get("container_id")
    if container_id is None:
        return {"status": "awaiting_container", "run_id": run_id}
    if not isinstance(container_id, str) or not re.fullmatch(r"[a-f0-9]{64}", container_id):
        raise ValueError("Monitor requires a trusted Docker container ID")
    path = config.state_dir / "records" / f"{run_id}.manifest.json"
    if not path.exists():
        state = _inspect_container(container_id)
        if state["Status"] not in {"exited", "dead"}:
            return {"status": "running", "container_status": state["Status"], "run_id": run_id}
        usage = _seal_when_idle(config.usage_dir, run_id)
        if usage is None:
            return {"status": "draining", "run_id": run_id}
        _save(path, _record(config, run_id, container_id, state, usage))
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["container_id"] != container_id or record["params"]["model_run_id"] != run_id:
        raise ValueError("Terminal research record does not match its launch")
    exports = {}
    for operator, name in ((False, "shared"), (True, "operator")):
        try:
            exports[name] = export_record(
                path, config.state_dir / "manifest-queues" / name, operator=operator,
            )
        except Exception as error:
            exports[name] = {"status": "error", "error_type": type(error).__name__}
    return {"status": record["status"].lower(), "run_id": run_id, "tracking": exports,
            "trace_complete": record["trace_complete"]}


def monitor_once(config: Settings) -> dict[str, Any]:
    config.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.state_dir / "monitor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = _monitor(config)
        _save(config.state_dir / "progress.json", result)
        return result


def flush_pending(config: Settings) -> list[dict[str, Any]]:
    results = []
    for operator, directories in (
        (False, (*config.shared_queues, config.state_dir / "manifest-queues" / "shared")),
        (True, (*config.operator_queues, config.state_dir / "manifest-queues" / "operator")),
    ):
        for directory in directories:
            try:
                flushed = flush_queue(directory, operator=operator, continue_on_error=True)
                errors = sum(item["status"] == "error" for item in flushed)
                results.append({
                    "queue": str(directory), "operator": operator,
                    "status": "error" if errors else "ok", "errors": errors,
                    "pending": sum(item["status"] != "exported" for item in flushed),
                })
            except Exception as error:
                results.append({"queue": str(directory), "operator": operator, "status": "error",
                                "error_type": type(error).__name__})
    _save(config.state_dir / "flush-progress.json", {"queues": results})
    return results


def cycle(config: Settings, *, monitor: bool = True) -> dict[str, Any]:
    result = {}
    if monitor:
        try:
            result["monitor"] = monitor_once(config)
        except Exception as error:
            result["monitor"] = {"status": "monitor_error", "error_type": type(error).__name__}
            try:
                _save(config.state_dir / "progress.json", result["monitor"])
            except OSError as error:
                result["progress_error_type"] = type(error).__name__
    try:
        result["queues"] = flush_pending(config)
    except Exception as error:
        result["flush_error_type"] = type(error).__name__
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("once", "watch", "flush"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    config = Settings.load(args.config)
    while True:
        print(json.dumps(cycle(config, monitor=args.command != "flush")), flush=True)
        if args.command != "watch" and not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
