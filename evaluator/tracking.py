import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

_PUBLIC_PARAMS = {
    "campaign_id", "case_id", "candidate_id", "candidate_digest", "revision", "parent_ids",
    "proposal_index", "hypothesis", "change", "model", "harness", "harness_version", "parameters",
    "metric_version", "metric_package_digest", "direction", "dataset_id", "dataset_version",
    "conditions_id",
}
_SECRET_KEY = re.compile(
    r"(^|_)(authorization|api_key|password|secret|access_token|refresh_token|capability|token)$",
    re.IGNORECASE,
)
_EXPORT_LOCK = RLock()


def redact(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            redact(str(key), secrets=secrets): (
                "[REDACTED]" if _SECRET_KEY.search(str(key).replace("-", "_"))
                else redact(item, secrets=secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, secrets=secrets) for item in value]
    if not isinstance(value, str):
        return value
    for secret in secrets:
        if secret:
            pattern = "".join(
                f"(?:{re.escape(char)}|\\\\u{ord(char):04x}|%{ord(char):02x})" for char in secret
            )
            value = re.sub(pattern, "[REDACTED]", value, flags=re.IGNORECASE)
    value = value.replace("\\/", "/")
    value = re.sub(r"(/runs/[^/\s]+/)[^/\s]+(/v1)", r"\1[REDACTED]\2", value)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[^\s\"',;]+", "[REDACTED]", value)
    value = re.sub(r"(https?://)[^/\s@]+@", r"\1[REDACTED]@", value)
    value = re.sub(
        r"(?i)(\b(?:api[_-]?key|access_token|password|authorization|secret)[\"']?"
        r"\s*[=:]\s*[\"']?)[^\s&,;\"']+", r"\1[REDACTED]", value,
    )
    return value


def sanitize_record(record: dict[str, Any], *, operator: bool = False) -> dict[str, Any]:
    split = record.get("split")
    if split not in {"train", "public", "private"}:
        raise ValueError("Unknown tracking split")
    if record.get("status") not in {"FINISHED", "FAILED"}:
        raise ValueError("Only completed records can be exported")
    if split == "private" and not operator:
        raise ValueError("Private records require operator export")
    if not isinstance(record.get("params"), dict) or not isinstance(record.get("metrics"), dict):
        raise ValueError("Tracking record requires params and metrics")
    if split == "public" and not operator:
        score = record["metrics"].get("score")
        if score is not None and (type(score) not in {int, float} or not math.isfinite(score)):
            raise ValueError("Public score must be a finite numeric scalar")
        record = {
            "split": "public", "status": record["status"],
            "params": {key: value for key, value in record["params"].items()
                       if key in _PUBLIC_PARAMS},
            "metrics": {"score": record["metrics"]["score"]}
            if record["metrics"].get("score") is not None else {},
        }
    return redact(record, secrets=_environment_secrets())


def _environment_secrets() -> tuple[str, ...]:
    with _EXPORT_LOCK:
        return tuple(
            value for key, value in os.environ.items()
            if _SECRET_KEY.search(key) and len(value) >= 8
        )


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    try:
        temporary.touch(mode=0o600, exist_ok=False)
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _destination(operator: bool) -> str:
    with _EXPORT_LOCK:
        shared = os.getenv("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
        if not operator:
            if not shared:
                raise ValueError("Configure the shared tracking destination explicitly")
            _validate_uri(shared)
            return shared
        target = os.getenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "").strip().rstrip("/")
        if not target:
            raise ValueError("Configure the operator tracking destination explicitly")
        if target == shared:
            raise ValueError("The operator tracking destination must be distinct")
        _validate_uri(target)
        return target


def _validate_uri(uri: str) -> None:
    parts = urlsplit(uri)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Tracking requires an HTTP(S) service")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Tracking URL must not contain credentials or query parameters")


@contextmanager
def _credentials(operator: bool):
    names = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN")
    previous = {key: os.environ.get(key) for key in names}
    try:
        if operator:
            for key in names:
                value = os.getenv(f"ARENA_OPERATOR_{key}")
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _tracking_uri(mlflow: Any, uri: str):
    previous_uri = mlflow.get_tracking_uri()
    previous_environment = os.environ.get("MLFLOW_TRACKING_URI")
    try:
        mlflow.set_tracking_uri(uri)
        yield
    finally:
        mlflow.set_tracking_uri(previous_uri)
        if previous_environment is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_environment


def _publish(record: dict[str, Any], uri: str, export_id: str, operator: bool) -> str:
    mlflow = importlib.import_module("mlflow")
    with _EXPORT_LOCK, _credentials(operator), _tracking_uri(mlflow, uri):
        client = mlflow.MlflowClient(tracking_uri=uri)
        name = f"arena-{'operator' if operator else record['split']}/{record['params']['case_id']}"
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        existing = client.search_runs(
            [experiment_id], filter_string=f"tags.arena_export_id = '{export_id}'", max_results=1,
        )
        if existing and existing[0].data.tags.get("arena_export_complete") == "true":
            return existing[0].info.run_id
        run_id = existing[0].info.run_id if existing else None
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id,
            run_name=str(record["params"]["candidate_id"]),
            tags={"arena_export_id": export_id},
        ) as run:
            run_id = run.info.run_id
            mlflow.set_tags({
                "split": record["split"], "campaign_id": record["params"]["campaign_id"],
                "arena_export_id": export_id,
                "arena_visibility": "operator" if operator else "shared",
            })
            mlflow.log_params({
                key: (json.dumps(value, ensure_ascii=False)
                      if isinstance(value, (dict, list)) else value)
                for key, value in record["params"].items() if value is not None
            })
            mlflow.log_metrics({
                key: value for key, value in record["metrics"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            })
            mlflow.log_dict(record, "record.json")
            if record.get("calls"):
                traces = client.search_traces(
                    locations=[experiment_id], run_id=run_id, max_results=1,
                )
                if not traces:
                    _spans(mlflow, record, run_id)
                    if not client.search_traces(
                        locations=[experiment_id], run_id=run_id, max_results=1,
                    ):
                        raise RuntimeError("Exported trace is not available from tracking server")
        client.set_terminated(run_id, status=record["status"])
        client.set_tag(run_id, "arena_export_complete", "true")
        return run_id


def _ordered_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = [{**call, "span_id": call.get("span_id", f"call-{index}")}
               for index, call in enumerate(calls)]
    identifiers = {call["span_id"] for call in pending}
    if len(identifiers) != len(pending):
        raise ValueError("Duplicate span ID")
    if any(call.get("parent_id") not in identifiers | {None} for call in pending):
        raise ValueError("Unknown parent span")
    ordered = []
    completed = {None}
    while pending:
        ready = [call for call in pending if call.get("parent_id") in completed]
        if not ready:
            raise ValueError("Cyclic span hierarchy")
        ordered.extend(ready)
        completed.update(call["span_id"] for call in ready)
        pending = [call for call in pending if call["span_id"] not in completed]
    return ordered


def _spans(mlflow: Any, record: dict[str, Any], run_id: str) -> None:
    calls = _ordered_calls(record["calls"])
    root = mlflow.start_span_no_context(
        f"{record['split']}_evaluation", inputs=record["params"],
        attributes={"run_id": run_id}, start_time_ns=record["start_time_ns"],
    )
    spans = {}
    for call in calls:
        attributes = call.get("attributes", {})
        span = mlflow.start_span_no_context(
            call["name"], parent_span=spans.get(call.get("parent_id"), root),
            span_type=call.get("span_type", "UNKNOWN"), inputs=call.get("inputs"),
            attributes=attributes, start_time_ns=call.get("start_time_ns", record["start_time_ns"]),
        )
        spans[call["span_id"]] = span
    for call in reversed(calls):
        spans[call["span_id"]].end(
            outputs=call.get("outputs"), status=call.get("status", "OK"),
            end_time_ns=call.get("end_time_ns", record["end_time_ns"]),
        )
    root.end(
        outputs=record["metrics"], status="OK" if record["status"] == "FINISHED" else "ERROR",
        end_time_ns=record["end_time_ns"],
    )
    mlflow.flush_trace_async_logging()


def _export(path: Path, entry: dict[str, Any]) -> dict[str, str]:
    if entry["status"] == "exported":
        return {"status": "exported", "run_id": entry["run_id"]}
    try:
        run_id = _publish(entry["record"], entry["uri"], entry["export_id"], entry["operator"])
    except Exception as error:
        entry.update(
            status="tracking_pending", error=redact(str(error), secrets=_environment_secrets()),
            attempts=entry.get("attempts", 0) + 1,
        )
        _save(path, entry)
        return {"status": "tracking_pending"}
    entry.update(status="exported", run_id=run_id)
    entry.pop("error", None)
    _save(path, entry)
    return {"status": "exported", "run_id": run_id}


def export_record(
    record_path: Path, queue_dir: Path | None = None, *, operator: bool = False,
) -> dict[str, str]:
    uri = _destination(operator)
    record = sanitize_record(json.loads(record_path.read_text(encoding="utf-8")), operator=operator)
    identity = json.dumps([uri, operator, record], sort_keys=True, ensure_ascii=False).encode()
    export_id = hashlib.sha256(identity).hexdigest()
    directory = queue_dir if queue_dir is not None else record_path.parent / ".tracking"
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = directory / f"{export_id}.json"
    with (directory / f"{export_id}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entry = json.loads(path.read_text()) if path.exists() else {
            "uri": uri, "operator": operator, "record": record, "export_id": export_id,
            "status": "tracking_pending",
        }
        if not path.exists():
            _save(path, entry)
        return _export(path, entry)


def export_configured(
    record_path: Path, queue_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    split = record.get("split")
    if split not in {"train", "public", "private"}:
        raise ValueError("Unknown tracking split")
    directory = queue_dir if queue_dir is not None else record_path.parent / ".tracking"
    with _EXPORT_LOCK:
        configured = {
            "shared": bool(os.getenv("MLFLOW_TRACKING_URI", "").strip()),
            "operator": bool(os.getenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "").strip()),
        }
    if split == "private" and configured["shared"] and not configured["operator"]:
        raise ValueError("Private export requires the operator tracking destination")
    results = {}
    for name, operator in (("shared", False), ("operator", True)):
        if configured[name] and (split != "private" or operator):
            results[name] = export_record(record_path, directory / name, operator=operator)
    return results


def flush_queue(
    directory: Path, *, operator: bool = False, continue_on_error: bool = False,
) -> list[dict[str, str]]:
    results = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.with_suffix(".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                entry = json.loads(path.read_text(encoding="utf-8"))
                if entry["operator"] != operator:
                    raise ValueError("Queue requires matching operator mode")
                entry["record"] = sanitize_record(entry["record"], operator=operator)
                if entry["uri"].rstrip("/") != _destination(operator):
                    raise ValueError("Queue destination differs from configured destination")
                if entry["status"] != "exported":
                    results.append(_export(path, entry))
        except Exception as error:
            if not continue_on_error:
                raise
            results.append({"status": "error", "error_type": type(error).__name__})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("record", type=Path)
    export.add_argument("--queue", type=Path)
    export.add_argument("--operator", action="store_true")
    flush = commands.add_parser("flush")
    flush.add_argument("--queue", type=Path, required=True)
    flush.add_argument("--operator", action="store_true")
    args = parser.parse_args()
    result = export_record(args.record, args.queue, operator=args.operator) \
        if args.command == "export" else flush_queue(args.queue, operator=args.operator)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
