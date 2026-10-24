import argparse
import asyncio
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from evaluator.build_receipts import (
    build_environment,
    load_build_receipt,
    validate_build_environment,
)
from evaluator.candidates import materialize_candidate
from evaluator.capacity import CapacityError, acquire_slot
from evaluator.client import pack_candidate, request
from evaluator.contracts import Candidate, Example, StrictModel
from evaluator.execution import collect_model_calls, make_runner
from evaluator.judge import ModelClient
from evaluator.metric_package import load_metric, metric_package_provenance
from evaluator.metrics import (
    CommandMetricError,
    MetricModelAccess,
    aggregate_results,
    evaluate_answer,
)
from evaluator.reliability import MODEL_RETRIES, add_usage, empty_usage, validate_timing
from evaluator.runner import HarnessRunner, RunnerError, SolveResult
from evaluator.search import Journal, Resources
from evaluator.service import (
    MAX_BODY_BYTES,
    RequestBoundary,
    Settings,
    _normalized_provenance,
    _usage,
    _write_json,
)


@dataclass(frozen=True)
class TrainSettings:
    evaluation: Settings
    campaign_id: str
    case_id: str
    dataset_id: str
    conditions_id: str
    tracking: bool = True

    @classmethod
    def from_env(cls) -> "TrainSettings":
        tracking = os.getenv("ARENA_TRAIN_TRACKING", "true") == "true"
        return cls(Settings.from_env(), tracking=tracking, **{
            name: os.environ["ARENA_" + name.upper()]
            for name in ("campaign_id", "case_id", "dataset_id", "conditions_id")
        })


class TrainSubmission(StrictModel):
    candidate: Candidate
    example_ids: list[str] | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _clean(value: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    encoded = re.sub(r"(/runs/[a-f0-9-]{36}/)[a-f0-9]{64}(/v1)", r"\1[redacted]\2", encoded)
    return json.loads(encoded)


def _save(path: Path, record: dict[str, Any]) -> None:
    _write_json(path, _clean(record))


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(record.get("metrics", {}))
    if record["status"] == "FAILED":
        metrics.pop("score", None)
    for role in ("solver", "judge"):
        resources = [item["resources"][role] for item in record["examples"]]
        if role == "solver" and record.get("preparation") is not None:
            resources.append(record["preparation"]["resources"]["solver"])
        if role == "judge" and "resources" in record.get("aggregation", {}):
            resources.append(record["aggregation"]["resources"]["judge"])
        metrics[f"{role}_tokens"] = sum(item["usage"]["total_tokens"] for item in resources)
        metrics[f"{role}_seconds"] = sum(item["wall_seconds"] for item in resources)
        metrics[f"{role}_unreported_attempts"] = sum(
            item["usage"]["unreported_attempts"] for item in resources
        )
        metrics[f"{role}_resources_complete"] = all(
            item.get("usage_complete", True) and item.get("wall_seconds_complete", True)
            for item in resources
        )
    metrics["wall_seconds"] = (time.time_ns() - record["start_time_ns"]) / 1e9
    return metrics


def _call(
    record: dict[str, Any], role: str, started: int, inputs: Any, outputs: Any,
    *, example_id: str | None = None, failed: bool = False,
) -> None:
    record["calls"].append({
        "name": role, "inputs": inputs, "outputs": outputs,
        "attributes": {"role": role, "example_id": example_id},
        "status": "ERROR" if failed else "OK", "start_time_ns": started,
        "end_time_ns": time.time_ns(),
    })


def create_app(
    configuration: TrainSettings,
    solve: Callable[[Path, str], SolveResult] | None = None,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> FastAPI:
    settings = configuration.evaluation
    examples = [Example.model_validate_json(line) for line in
                settings.dataset_path.read_bytes().splitlines() if line.strip()]
    by_id = {example.id: example for example in examples}
    if not examples or len(by_id) != len(examples) or any(not key.strip() for key in by_id):
        raise ValueError("train dataset must be nonempty with unique IDs")
    metric = load_metric(settings.metric_path)
    package = metric_package_provenance(settings.metric_path)
    if metric.kind == "llm" and any(not example.reference for example in examples):
        raise ValueError("LLM train metric requires references")
    root = settings.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    usage_dir = settings.usage_dir or root / "usage"
    build_dir = settings.build_dir
    slot_dir = settings.slot_dir
    slots = settings.execution_slots
    uses_model = any(command and command.uses_model
                     for command in (metric.command, metric.aggregation))
    trace_dir = settings.trace_dir

    def collect_calls(
        record: dict[str, Any], run_ids: list[str], role: str, example_id: str | None = None,
    ) -> None:
        calls, errors = collect_model_calls(
            trace_dir, run_ids, attributes={"role": role, "example_id": example_id},
        )
        record["calls"].extend(calls)
        if errors:
            record.setdefault("trace_errors", []).extend(errors)

    runner = solve or make_runner(settings, usage_dir)
    frozen = {
        "dataset": [item.model_dump() for item in examples], "metric": metric.model_dump(),
        "package": package, "model": settings.model, "parameters": settings.parameters,
        "runner": runner.provenance() if isinstance(runner, HarnessRunner) else "test",
        "campaign_id": configuration.campaign_id, "case_id": configuration.case_id,
        "dataset_id": configuration.dataset_id, "conditions_id": configuration.conditions_id,
    }

    def export_record(record_path: Path, record: dict[str, Any]) -> None:
        if not configuration.tracking:
            return
        try:
            exporter = importlib.import_module("evaluator.tracking").export_configured
            record["tracking"] = exporter(
                record_path, queue_dir=settings.tracking_queue_dir or root / "tracking",
            )
        except Exception as error:
            record["tracking"] = {"status": "tracking_pending",
                                  "error_type": type(error).__name__}
        _save(record_path, record)

    def recover_active(record: dict[str, Any]) -> None:
        stage = record.pop("active_stage", None)
        if stage is None:
            return
        role = stage["role"]
        if role == "aggregation":
            aggregation = record.setdefault("aggregation", {"status": "failed"})
            resource = aggregation.setdefault("resources", {}).setdefault(
                "judge", {"usage": empty_usage(), "wall_seconds": 0.0},
            )
        else:
            example_id = stage["example_id"]
            row = next((item for item in record["examples"]
                        if item["example_id"] == example_id), None)
            if row is None:
                answer = root / "answers" / record["params"]["candidate_digest"]
                answer = answer / (_digest(example_id) + ".json")
                if answer.exists():
                    record["examples"].append(json.loads(answer.read_text()))
                    return
                example = by_id[example_id]
                row = {"example_id": example_id, "input": example.input,
                       "reference": example.reference, "failure": "interrupted",
                       "resources": {
                           name: {"usage": empty_usage(), "wall_seconds": 0.0}
                           for name in ("solver", "judge")
                       }}
                record["examples"].append(row)
            resource = row["resources"][role]
        resource.update(usage_complete=False, wall_seconds_complete=False)

    with (root / "reservation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config_path = root / "configuration.json"
        if config_path.exists() and (
            _normalized_provenance(json.loads(config_path.read_text()))
            != _normalized_provenance(frozen)
        ):
            raise ValueError("train configuration is frozen")
        if not config_path.exists():
            _save(config_path, frozen)
        for record_path in root.glob("evaluations/*/record.json"):
            record = json.loads(record_path.read_text())
            if record["status"] == "RUNNING":
                candidate_lock = root / (record["params"]["candidate_digest"] + ".lock")
                with candidate_lock.open("a+b") as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if ("active_stage" not in record
                            and record.get("aggregation", {}).get("status") == "complete"):
                        record.update(status="FINISHED", end_time_ns=time.time_ns())
                    else:
                        recover_active(record)
                        record.update(status="FAILED", error="interrupted",
                                      end_time_ns=time.time_ns())
                    record["metrics"] = _metrics(record)
                    _save(record_path, record)
                    export_record(record_path, record)

    def persist(record: dict[str, Any], path: Path) -> None:
        record["metrics"] = _metrics(record)
        completed = sum("score" in item or "statistics" in item for item in record["examples"])
        total = len(record["example_ids"])
        elapsed = record["metrics"]["wall_seconds"]
        record["progress"] = {
            "completed_examples": completed, "total_examples": total,
            "elapsed_seconds": elapsed,
            "remaining_seconds": elapsed / completed * (total - completed) if completed else None,
        }
        _save(path, record)

    def evaluate(record: dict[str, Any], candidate_dir: Path, record_path: Path) -> None:
        try:
            model_access = MetricModelAccess(
                settings.model, settings.runner_model_base_url, usage_dir,
                settings.model_timeout, settings.retry_delay,
            ) if uses_model else None

            def model_runs() -> list[str]:
                return getattr(model_access, "run_ids", [])

            judge_usage = empty_usage()
            judge_complete = complete
            judge_client = None
            judge_calls: list[dict[str, Any]] = []
            if metric.kind == "llm":
                if complete is None:
                    judge_client = ModelClient(
                        settings.model_base_url, settings.model, settings.model_api_key,
                        settings.parameters, timeout=settings.model_timeout,
                        retries=MODEL_RETRIES, retry_delay=settings.retry_delay,
                        on_attempt=judge_calls.append,
                    )
                    judge_complete = judge_client
                    judge_usage = judge_client.usage
                else:
                    callback = complete

                    def tracked(messages: list[dict[str, str]]) -> str:
                        add_usage(judge_usage, None)
                        return callback(messages)
                    judge_complete = tracked
            answer_dir = root / "answers" / record["params"]["candidate_digest"]
            answer_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            for example_id in record["example_ids"]:
                example = by_id[example_id]
                answer_path = answer_dir / (_digest(example_id) + ".json")
                started = time.time_ns()
                if answer_path.exists():
                    row = json.loads(answer_path.read_text())
                else:
                    row = {
                        "example_id": example_id, "input": example.input,
                        "reference": example.reference, "resources": {
                            role: {"usage": empty_usage(), "wall_seconds": 0.0}
                            for role in ("solver", "judge")
                        },
                    }
                    record["active_stage"] = {"role": "solver", "example_id": example_id,
                                              "start_time_ns": started}
                    persist(record, record_path)
                    try:
                        solved = runner(candidate_dir, example.input)
                        if (not isinstance(solved, SolveResult)
                                or not isinstance(solved.answer, str)):
                            raise ValueError("solver must return a measured answer")
                        row["answer"] = solved.answer
                        row["resources"]["solver"] = {
                            "usage": _usage(solved.usage), "wall_seconds": validate_timing(
                                solved.wall_seconds, "solver time", allow_zero=True,
                            ),
                        }
                        row["run_id"] = getattr(solved, "run_id", None)
                        row["resources"]["solver"]["run_id"] = row["run_id"]
                        _save(answer_path, row)
                    except Exception as error:
                        failed_usage = error.usage if isinstance(error, RunnerError) else None
                        measured = failed_usage is not None
                        row["resources"]["solver"] = {
                            "usage": _usage(failed_usage) if failed_usage is not None
                            else empty_usage(),
                            "wall_seconds": error.wall_seconds
                            if isinstance(error, RunnerError) and error.wall_seconds is not None
                            else (time.time_ns() - started) / 1e9,
                            "usage_complete": measured,
                        }
                        row["failure"] = "solver_failure"
                        record["examples"].append(row)
                        _call(record, "solver", started, example.input,
                              {"error": type(error).__name__}, example_id=example_id, failed=True)
                        if isinstance(error, RunnerError) and error.run_id:
                            collect_calls(record, [error.run_id], "solver", example_id)
                        raise
                _call(record, "solver", started, example.input, row["answer"],
                      example_id=example_id)
                record["examples"].append(row)
                record.pop("active_stage", None)
                if row.get("run_id"):
                    collect_calls(record, [row["run_id"]], "solver", example_id)
                persist(record, record_path)
                if "score" in row or "statistics" in row:
                    collect_calls(record, row["resources"]["judge"].get("run_ids", []),
                                  "judge", example_id)
                    record["calls"].extend(row.get("judge_calls", []))
                    if row.get("judge_trace_errors"):
                        record.setdefault("trace_errors", []).extend(row["judge_trace_errors"])
                    continue
                before = judge_usage.copy()
                before_runs = len(model_runs())
                judge_calls.clear()
                if judge_client is not None:
                    judge_client.trace_errors.clear()
                started = time.time_ns()
                record["active_stage"] = {"role": "judge", "example_id": example_id,
                                          "start_time_ns": started}
                persist(record, record_path)
                try:
                    outcome = evaluate_answer(
                        metric, example, row["answer"], judge_complete,
                        retries=MODEL_RETRIES, retry_delay=settings.retry_delay,
                        model_access=model_access, usage=judge_usage,
                    )
                    row.update(outcome)
                except Exception as error:
                    row["failure"] = "judge_failure"
                    if isinstance(error, CommandMetricError):
                        row["diagnostics"] = error.diagnostics
                    raise
                finally:
                    if judge_client is not None:
                        row["judge_calls"] = [{
                            **call, "attributes": {
                                **call["attributes"], "role": "judge", "example_id": example_id,
                            },
                        } for call in judge_calls]
                        record["calls"].extend(row["judge_calls"])
                        if judge_client.trace_errors:
                            row["judge_trace_errors"] = [{
                                **error, "role": "judge", "example_id": example_id,
                            } for error in judge_client.trace_errors]
                            record.setdefault("trace_errors", []).extend(row["judge_trace_errors"])
                    row["resources"]["judge"] = {
                        "usage": {key: judge_usage[key] - before[key] for key in before},
                        "wall_seconds": (time.time_ns() - started) / 1e9,
                        "usage_complete": row.get("diagnostics", {}).get(
                            "model_usage_complete", True,
                        ),
                        "run_ids": model_runs()[before_runs:],
                    }
                    collect_calls(record, model_runs()[before_runs:], "judge", example_id)
                    _call(record, "judge", started, {"answer": row["answer"]},
                          {key: value for key, value in row.items() if key in {
                              "score", "statistics", "failure", "diagnostics",
                          }}, example_id=example_id, failed="failure" in row)
                    record.pop("active_stage", None)
                    persist(record, record_path)
                    if "failure" not in row:
                        _save(answer_path, row)
            before = judge_usage.copy()
            before_runs = len(model_runs())
            started = time.time_ns()
            aggregation: dict[str, Any] = {"status": "failed"}
            record["aggregation"] = aggregation
            record["active_stage"] = {"role": "aggregation", "start_time_ns": started}
            persist(record, record_path)
            try:
                score = aggregate_results(
                    metric, record["examples"], model_access=model_access, usage=judge_usage,
                )
                aggregation.update(status="complete", score=score)
                record["metrics"]["score"] = score
            except CommandMetricError as error:
                aggregation["diagnostics"] = error.diagnostics
                raise
            finally:
                aggregation["resources"] = {"judge": {
                    "usage": {key: judge_usage[key] - before[key] for key in before},
                    "wall_seconds": (time.time_ns() - started) / 1e9,
                    "usage_complete": aggregation.get("diagnostics", {}).get(
                        "model_usage_complete", True,
                    ),
                    "run_ids": model_runs()[before_runs:],
                }}
                collect_calls(record, model_runs()[before_runs:], "judge")
                _call(record, "aggregation", started, record["example_ids"], aggregation,
                      failed=aggregation["status"] != "complete")
                record.pop("active_stage", None)
                persist(record, record_path)
            record["status"] = "FINISHED"
        except Exception as error:
            record.update(status="FAILED", error={
                "type": type(error).__name__, "message": str(error)[:2000],
            })
        finally:
            record.pop("active_stage", None)
            record["end_time_ns"] = time.time_ns()
            persist(record, record_path)
            export_record(record_path, record)

    def execute(
        record: dict[str, Any], candidate_dir: Path, record_path: Path,
        lease: BinaryIO, slot: BinaryIO | None, completion: Future[None], started: threading.Event,
    ) -> None:
        started.set()
        try:
            evaluate(record, candidate_dir, record_path)
        except BaseException as error:
            completion.set_exception(error)
        else:
            completion.set_result(None)
        finally:
            lease.close()
            if slot is not None:
                slot.close()

    async def wait_for_evaluation(completion: Future[None]) -> None:
        await asyncio.shield(asyncio.wrap_future(completion))

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestBoundary, token=settings.token,
                       max_body_bytes=settings.max_body_bytes or MAX_BODY_BYTES)
    staging = settings.asset_staging_dir
    store = settings.asset_store_dir
    if staging and store:
        install_assets = importlib.import_module("evaluator.assets").install_asset_routes
        install_assets(app, Path(staging), Path(store))
    if build_dir is not None and isinstance(runner, HarnessRunner):
        from evaluator.build import BuildStore, install_build_routes, run_build

        published = settings.published_assets_dir
        if published is None:
            raise ValueError("trusted builds require published asset storage")

        def export_build(receipt: dict[str, Any]) -> None:
            stage = receipt["own_resources"]["solver"]
            record = {
                "split": "train", "status": receipt["status"],
                "start_time_ns": receipt["start_time_ns"], "end_time_ns": receipt["end_time_ns"],
                "params": {
                    "campaign_id": configuration.campaign_id, "case_id": configuration.case_id,
                    "candidate_id": receipt["candidate_id"],
                    "candidate_digest": receipt["candidate_digest"],
                    "model": settings.model, "measurement_kind": "preparation",
                    "subset_id": "build", "conditions_id": configuration.conditions_id,
                    "input_assets_digest": receipt["input_assets_digest"],
                    "environment_fingerprint": receipt["environment_fingerprint"],
                },
                "metrics": {"solver_tokens": stage["usage"]["total_tokens"],
                            "solver_seconds": stage["wall_seconds"],
                            "solver_unreported_attempts": stage["usage"]["unreported_attempts"],
                            "judge_tokens": 0, "judge_seconds": 0, "judge_unreported_attempts": 0},
                "calls": receipt.get("own_calls", []), "preparation": receipt,
            }
            path = root / "build-records" / (receipt["candidate_digest"] + ".json")
            _save(path, record)
            export_record(path, record)

        if settings.input_assets_digest is None:
            raise ValueError("trusted builds require an input assets digest")
        build_store = BuildStore(
            build_dir, published, settings.input_assets_digest,
            build_environment(runner, settings.parameters),
            lambda candidate, output: run_build(runner, candidate, output),
            slot_dir=slot_dir, slots=slots, trace_dir=trace_dir, on_complete=export_build,
            dependency_assets=runner.build_asset_digests,
        )
        validate_build_environment(build_dir, build_environment(runner, settings.parameters),
                                   settings.input_assets_digest)
        install_build_routes(app, build_store)

    @app.exception_handler(RequestValidationError)
    async def invalid_submission(request: Any, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": "invalid submission"}, status_code=422)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/evaluations/{evaluation_id}")
    def status(evaluation_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[a-f0-9]{64}", evaluation_id):
            raise HTTPException(404, "no evaluation")
        path = root / "evaluations" / evaluation_id / "record.json"
        if not path.exists():
            raise HTTPException(404, "no evaluation")
        return json.loads(path.read_text())

    @app.post("/v1/evaluations")
    def submit(submission: TrainSubmission, background: BackgroundTasks) -> JSONResponse:
        ids = submission.example_ids if submission.example_ids is not None else list(by_id)
        if len(ids) != len(set(ids)) or set(ids) - set(by_id):
            raise HTTPException(422, "invalid train subset")
        ids = [key for key in by_id if key in ids]
        candidate = submission.candidate
        digest = candidate.digest()
        evaluation_id = _digest([digest, ids])
        run_dir = root / "evaluations" / evaluation_id
        path = run_dir / "record.json"
        with (root / "reservation.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                existing = json.loads(path.read_text())
                return JSONResponse({
                    "evaluation_id": evaluation_id, "status": existing["status"],
                }, status_code=200)
            lease = (root / (digest + ".lock")).open("a+b")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                raise HTTPException(409, "candidate is already running") from None
            slot = None
            package_dir = None
            try:
                try:
                    slot = acquire_slot(slot_dir, slots)
                except CapacityError:
                    raise HTTPException(503, "evaluation capacity is busy",
                                        headers={"Retry-After": "60"}) from None
                run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                package_dir = Path(tempfile.mkdtemp(prefix="package-", dir=run_dir))
                candidate_dir = package_dir / "candidate"
                materialize_candidate(candidate, candidate_dir)
                try:
                    if isinstance(runner, HarnessRunner):
                        runner.validate_candidate(candidate_dir)
                        preparation = runner.preparation_receipt(candidate_dir, digest)
                    elif settings.runner_supports_build:
                        preparation = load_build_receipt(build_dir, candidate_dir, digest)
                    else:
                        preparation = None
                except (OSError, ValueError, RunnerError):
                    raise HTTPException(422, "candidate preparation is unavailable") from None
                manifest = candidate_dir / "solution.json"
                try:
                    selection = json.loads(manifest.read_text()) if manifest.exists() else {}
                except (ValueError, UnicodeError):
                    selection = {}
                if not isinstance(selection, dict):
                    selection = {}
                allowed = {"revision", "parent_ids", "proposal_index", "hypothesis", "change",
                           "proposed_at", "measurement_id", "decision"}
                metadata = {key: value for key, value in submission.metadata.items()
                            if key in allowed}
                record = {
                    "split": "train", "evaluation_id": evaluation_id, "status": "RUNNING",
                    "example_ids": ids, "examples": [], "metrics": {},
                    "preparation": preparation,
                    "calls": list(preparation.get("calls", [])) if preparation else [],
                    "start_time_ns": time.time_ns(),
                    "params": {
                        **metadata, "campaign_id": configuration.campaign_id,
                        "case_id": configuration.case_id, "candidate_id": candidate.id,
                        "candidate_digest": digest, "dataset_id": configuration.dataset_id,
                        "subset_id": _digest(ids), "conditions_id": configuration.conditions_id,
                        "metric_version": metric.version, "direction": metric.direction,
                        "model": settings.model, "model_parameters": settings.parameters,
                        "harness": selection.get("harness", "default"),
                        "parameters": selection.get("parameters", {}),
                        "harness_version": settings.runner_image,
                    },
                }
                persist(record, path)
                completion: Future[None] = Future()
                started = threading.Event()
                thread = threading.Thread(target=execute, args=(
                    record, candidate_dir, path, lease, slot, completion, started,
                ), daemon=True)
                thread.start()
                started.wait()
            except BaseException:
                lease.close()
                if slot is not None:
                    slot.close()
                if package_dir is not None and not path.exists():
                    shutil.rmtree(package_dir)
                raise
        background.add_task(wait_for_evaluation, completion)
        return JSONResponse({"evaluation_id": evaluation_id, "status": "RUNNING"}, status_code=202)

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trusted train service and researcher client")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    submit = commands.add_parser("submit")
    submit.add_argument("--candidate", required=True)
    submit.add_argument("--measurement", required=True)
    submit.add_argument("--dataset", required=True)
    submit.add_argument("--conditions", required=True)
    submit.add_argument("--example-id", action="append")
    submit.add_argument("--train-data", type=Path, required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--evaluation-id", required=True)
    result_parser = commands.add_parser("result")
    result_parser.add_argument("--evaluation-id", required=True)
    result_parser.add_argument("--measurement", required=True)
    result_parser.add_argument("--output", type=Path, required=True)
    for command in (submit, status_parser, result_parser):
        command.add_argument("--url", required=True)
        command.add_argument("--token-file", type=Path, required=True)
    for command in (submit, result_parser):
        command.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn
        uvicorn.run(create_app(TrainSettings.from_env()), host=args.host, port=args.port,
                    access_log=False)
        return 0
    if args.command == "submit":
        journal = Journal(args.journal)
        proposals = journal.export()["candidates"]
        proposal = next(item for item in proposals if item["candidate_id"] == args.candidate)
        ids = [Example.model_validate_json(line).id for line in
               args.train_data.read_bytes().splitlines() if line.strip()]
        if args.example_id is not None:
            if len(args.example_id) != len(set(args.example_id)) or set(args.example_id) - set(ids):
                raise ValueError("invalid train subset")
            ids = [key for key in ids if key in args.example_id]
        with tempfile.TemporaryDirectory() as temporary:
            restored = Path(temporary) / "candidate"
            journal.restore(args.candidate, restored)
            candidate = pack_candidate(args.candidate, restored)
        comparison = {"split": "train", "dataset_id": args.dataset,
                      "subset_id": _digest(ids), "conditions_id": args.conditions}
        previous = next((item for item in journal.export()["measurements"]
                         if item["measurement_id"] == args.measurement), None)
        if previous is None:
            journal.begin(args.measurement, args.candidate, **comparison, job=args.url)
        elif (previous["candidate_id"] != args.candidate or previous["comparison"] != comparison
              or previous["state"] != "running"):
            raise ValueError("measurement already exists with different inputs or finished")
        if journal.status()["stopped"]:
            raise ValueError("campaign stopped; retrieve existing checks with status or result")
        result = request("POST", args.url, args.token_file, {
            "candidate": candidate.model_dump(), "example_ids": ids, "metadata": {
                "parent_ids": proposal["parents"], "proposal_index": proposal["proposal"],
                "hypothesis": proposal["hypothesis"], "change": proposal["change"],
                "proposed_at": proposal["created_at"], "measurement_id": args.measurement,
            },
        })
    else:
        result: dict[str, Any] = cast(dict[str, Any], request(
            "GET", args.url, args.token_file, evaluation_id=args.evaluation_id,
        ))
        if args.command == "result":
            _save(args.output, result)
            if result["status"] not in {"FINISHED", "FAILED"}:
                print(json.dumps({"status": result["status"], "saved": str(args.output)}))
                return 0
            metrics = result["metrics"]
            journal = Journal(args.journal)
            measurement = next(item for item in journal.export()["measurements"]
                               if item["measurement_id"] == args.measurement)
            if measurement["digest"] != result["params"]["candidate_digest"]:
                raise ValueError("result candidate differs from journal")
            comparison = {"split": "train", **{
                key: result["params"][key]
                for key in ("dataset_id", "subset_id", "conditions_id")
            }}
            if measurement["comparison"] != comparison:
                raise ValueError("result comparison differs from journal")
            if measurement["state"] == "running":
                journal.finish(args.measurement, **{
                    role: Resources(metrics[f"{role}_tokens"], metrics[f"{role}_seconds"],
                                    metrics[f"{role}_unreported_attempts"] == 0
                                    and metrics[f"{role}_resources_complete"])
                    for role in ("solver", "judge")
                }, score=metrics.get("score"),
                    error="evaluation_failed" if result["status"] == "FAILED" else None,
                    evidence={"record": str(args.output), "evaluation_id": args.evaluation_id})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
