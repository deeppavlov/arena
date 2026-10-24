import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from evaluator.build_receipts import (
    build_environment,
    load_build_receipt,
    validate_build_environment,
)
from evaluator.candidates import materialize_candidate
from evaluator.capacity import CapacityError, acquire_slot
from evaluator.contracts import Example, Submission
from evaluator.execution import collect_model_calls, make_runner, solver_metadata
from evaluator.judge import ModelClient
from evaluator.metric_package import load_metric, metric_package_provenance
from evaluator.metrics import (
    CommandMetricError,
    MetricModelAccess,
    aggregate_results,
    command_provenance,
    evaluate_answer,
)
from evaluator.ranking import RANKING_POLICY, choose_winners
from evaluator.reliability import (
    MODEL_RETRIES,
    MODEL_TIMEOUT,
    PREPARE_TIMEOUT,
    RETRY_DELAY,
    RUNNER_TIMEOUT,
    add_usage,
    empty_usage,
    parse_usage,
    validate_timing,
)
from evaluator.runner import (
    RunnerError,
    SolveResult,
    validate_runner_resources,
)
from evaluator.search import Journal, SearchError
from evaluator.tracking import export_configured

MAX_BODY_BYTES = 70 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    metric_path: Path
    state_dir: Path
    token: str = field(repr=False)
    runner_image: str
    release_id: str
    model: str = ""
    model_base_url: str = ""
    model_api_key: str = field(default="", repr=False)
    runner_network: str = "none"
    runner_model_base_url: str = "http://model-gateway:8081/v1"
    runner_timeout: float = RUNNER_TIMEOUT
    model_timeout: float = MODEL_TIMEOUT
    prepare_timeout: float = PREPARE_TIMEOUT
    retry_delay: float = RETRY_DELAY
    usage_dir: Path | None = None
    assets_dir: Path | None = None
    finalists: int = 5
    output_count: int = 1
    split: str = "private"
    uses_model: bool = True
    harnesses: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_body_bytes: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    runner_memory: str = "2g"
    runner_cpus: str = "2"
    runner_pids_limit: int = 128
    runner_tmpfs_size: str = "128m"
    runner_tmpfs_exec: bool = False
    trace_dir: Path | None = None
    tracking_dir: Path | None = None
    tracking_queue_dir: Path | None = None
    case_id: str = ""
    slot_dir: Path | None = None
    execution_slots: int = 4
    published_assets_dir: Path | None = None
    build_dir: Path | None = None
    input_assets_digest: str | None = None
    runner_supports_build: bool = False
    runner_supports_asset_selection: bool = False
    asset_staging_dir: Path | None = None
    asset_store_dir: Path | None = None
    search_mode: str = "standalone"
    search_journal: Path | None = None
    search_max_solutions: int | None = None
    search_stop_path: Path | None = None

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.token, self.runner_image, self.release_id)):
            raise ValueError("token, runner image and release ID are required")
        if type(self.finalists) is not int or self.finalists < 1:
            raise ValueError("finalists must be a positive integer")
        if type(self.output_count) is not int or not 1 <= self.output_count <= self.finalists:
            raise ValueError("output_count must be between 1 and finalists")
        if self.split not in {"private", "public"}:
            raise ValueError("split must be private or public")
        if self.search_mode not in {"standalone", "campaign"}:
            raise ValueError("search mode must be standalone or campaign")
        if self.search_mode == "campaign" and self.search_journal is None:
            raise ValueError("search campaign requires a journal and max_solutions")
        if self.search_mode == "standalone" and self.search_journal is not None:
            raise ValueError("search journal requires campaign mode")
        if (self.search_journal is None) != (self.search_max_solutions is None):
            raise ValueError("search journal and max_solutions must be configured together")
        if self.search_max_solutions is not None and (
            type(self.search_max_solutions) is not int or self.search_max_solutions < 1
        ):
            raise ValueError("search max_solutions must be a positive integer")
        if self.search_stop_path is not None and self.search_journal is None:
            raise ValueError("search stop path requires a search journal")
        for name in ("uses_model", "runner_supports_build", "runner_supports_asset_selection"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if (self.asset_staging_dir is None) != (self.asset_store_dir is None):
            raise ValueError("asset staging and store directories must be configured together")
        if self.max_body_bytes is not None and (
            type(self.max_body_bytes) is not int or self.max_body_bytes < 1
        ):
            raise ValueError("max_body_bytes must be a positive integer")
        if not isinstance(self.harnesses, dict):
            raise ValueError("harnesses must be a JSON object")
        for name, config in self.harnesses.items():
            if (
                not isinstance(name, str) or not name.strip()
                or not isinstance(config, dict)
                or set(config) - {"image", "uses_model", "network",
                                  "supports_build", "supports_asset_selection"}
                or not isinstance(config.get("image"), str) or not config["image"].strip()
                or type(config.get("uses_model", self.uses_model)) is not bool
                or type(config.get("supports_build", False)) is not bool
                or type(config.get("supports_asset_selection", False)) is not bool
                or not isinstance(config.get("network", self.runner_network), str)
            ):
                raise ValueError("invalid harness registry")
        if (self.uses_model or any(
            config.get("uses_model", self.uses_model) for config in self.harnesses.values()
        )) and (not self.model.strip() or not self.model_base_url.strip()):
            raise ValueError("model and endpoint are required for model-based harnesses")
        for name in ("runner_timeout", "model_timeout", "prepare_timeout"):
            validate_timing(getattr(self, name), name)
        validate_timing(self.retry_delay, "retry_delay", allow_zero=True)
        validate_runner_resources(
            self.runner_memory, self.runner_cpus, self.runner_pids_limit,
            self.runner_tmpfs_size, self.runner_tmpfs_exec,
        )
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a JSON object")
        if self.tracking_dir is not None and not self.case_id.strip():
            raise ValueError("case_id is required for tracking export")
        if type(self.execution_slots) is not int or self.execution_slots < 1:
            raise ValueError("execution_slots must be a positive integer")

    @classmethod
    def from_env(cls) -> "Settings":
        def value(name: str, default: str | None = None) -> str:
            name = "ARENA_" + name
            path = os.environ.get(name + "_FILE")
            if path:
                return Path(path).read_text().strip()
            result = os.environ.get(name, default)
            if result is None:
                raise ValueError(f"{name} is required")
            return result

        assets = value("ASSETS_DIR", "")
        usage = value("USAGE_DIR", "")
        trace = value("TRACE_DIR", "")
        tracking = value("TRACKING_DIR", "")
        tracking_queue = value("TRACKING_QUEUE_DIR", "")
        slots = value("SLOT_DIR", "")
        published_assets = value("PUBLISHED_ASSETS_DIR", "")
        builds = value("BUILD_DIR", "")
        asset_staging = value("ASSET_STAGING_DIR", "")
        asset_store = value("ASSET_STORE_DIR", "")
        search_journal = value("SEARCH_JOURNAL", "")
        search_max_solutions = value("SEARCH_MAX_SOLUTIONS", "")
        search_stop = value("SEARCH_STOP_PATH", "")
        capabilities = {}
        for name in ("RUNNER_SUPPORTS_BUILD", "RUNNER_SUPPORTS_ASSET_SELECTION"):
            enabled = value(name, "false").lower()
            if enabled not in {"true", "false"}:
                raise ValueError(f"ARENA_{name} must be true or false")
            capabilities[name.lower()] = enabled == "true"
        uses_model = value("USES_MODEL", "true").lower()
        if uses_model not in {"true", "false"}:
            raise ValueError("ARENA_USES_MODEL must be true or false")
        tmpfs_exec = value("RUNNER_TMPFS_EXEC", "false").lower()
        if tmpfs_exec not in {"true", "false"}:
            raise ValueError("ARENA_RUNNER_TMPFS_EXEC must be true or false")
        body_bytes = value("MAX_BODY_BYTES", "")
        return cls(
            dataset_path=Path(value("DATASET_PATH")),
            metric_path=Path(value("METRIC_PATH")),
            state_dir=Path(value("STATE_DIR")),
            token=value("TOKEN"),
            model=value("MODEL", ""),
            model_base_url=value("MODEL_BASE_URL", ""),
            runner_image=value("RUNNER_IMAGE"),
            release_id=value("RELEASE_ID"),
            model_api_key=value("MODEL_API_KEY", ""),
            runner_network=value("RUNNER_NETWORK", "none"),
            runner_model_base_url=value("RUNNER_MODEL_BASE_URL", "http://model-gateway:8081/v1"),
            runner_timeout=float(value("RUNNER_TIMEOUT", str(RUNNER_TIMEOUT))),
            model_timeout=float(value("MODEL_TIMEOUT", str(MODEL_TIMEOUT))),
            prepare_timeout=float(value("PREPARE_TIMEOUT", str(PREPARE_TIMEOUT))),
            retry_delay=float(value("RETRY_DELAY", str(RETRY_DELAY))),
            usage_dir=Path(usage) if usage else None,
            assets_dir=Path(assets) if assets else None,
            finalists=int(value("FINALISTS", "5")),
            output_count=int(value("OUTPUT_COUNT", "1")),
            split=value("SPLIT", "private"),
            uses_model=uses_model == "true",
            harnesses=json.loads(value("HARNESSES", "{}")),
            max_body_bytes=int(body_bytes) if body_bytes else None,
            parameters=json.loads(value("PARAMETERS", "{}")),
            runner_memory=value("RUNNER_MEMORY", "2g"),
            runner_cpus=value("RUNNER_CPUS", "2"),
            runner_pids_limit=int(value("RUNNER_PIDS_LIMIT", "128")),
            runner_tmpfs_size=value("RUNNER_TMPFS_SIZE", "128m"),
            runner_tmpfs_exec=tmpfs_exec == "true",
            trace_dir=Path(trace) if trace else None,
            tracking_dir=Path(tracking) if tracking else None,
            tracking_queue_dir=Path(tracking_queue) if tracking_queue else None,
            case_id=value("CASE_ID", ""),
            slot_dir=Path(slots) if slots else None,
            execution_slots=int(value("EXECUTION_SLOTS", "4")),
            published_assets_dir=Path(published_assets) if published_assets else None,
            build_dir=Path(builds) if builds else None,
            input_assets_digest=value("INPUT_ASSETS_DIGEST", "") or None,
            asset_staging_dir=Path(asset_staging) if asset_staging else None,
            asset_store_dir=Path(asset_store) if asset_store else None,
            search_mode=value("SEARCH_MODE", "standalone"),
            search_journal=Path(search_journal) if search_journal else None,
            search_max_solutions=int(search_max_solutions) if search_max_solutions else None,
            search_stop_path=Path(search_stop) if search_stop else None,
            **capabilities,
        )


class RequestBoundary:
    def __init__(self, app: ASGIApp, token: str, max_body_bytes: int) -> None:
        self.app = app
        self.authorization = b"Bearer " + token.encode()
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/healthz":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        if not secrets.compare_digest(headers.get(b"authorization", b""), self.authorization):
            await JSONResponse({"detail": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        if scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = self.max_body_bytes + 1
        if length < 0 or length > self.max_body_bytes:
            await JSONResponse({"detail": "body too large"}, status_code=413)(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await JSONResponse({"detail": "body too large"}, status_code=413)(
                    scope, receive, send,
                )
                return
            if not message.get("more_body", False):
                break
        pending = True

        async def buffered_receive() -> Message:
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _public(record: dict[str, Any], split: str) -> dict[str, Any]:
    response = {"status": record["status"]}
    if split == "public":
        response["evaluation_id"] = record["batch_digest"]
        if record["status"] == "complete":
            response["scores"] = [
                {"candidate_id": item["candidate_id"], "score": item["score"]}
                for item in record["scores"]
            ]
    return response


def _usage(value: dict[str, int]) -> dict[str, int]:
    parsed = parse_usage(value)
    for key in ("attempts", "unreported_attempts"):
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise ValueError("usage must include nonnegative integer attempt counts")
        parsed[key] = count
    if parsed["unreported_attempts"] > parsed["attempts"]:
        raise ValueError("unreported attempts exceed total attempts")
    if parsed["attempts"] == 0 and parsed["total_tokens"]:
        raise ValueError("token usage requires a model attempt")
    return parsed


def _normalized_provenance(value: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(value)
    value.pop("dataset_records_digest", None)
    for key in ("metric_package", "package"):
        package = value.get(key)
        if package is not None:
            package.pop("sha256", None)
            for entry in package.get("files", {}).values():
                entry.pop("sha256", None)
    metric = value.get("metric", {})
    if metric.get("votes") == 3:
        metric.pop("votes")
    for section in (metric, value.get("metric_command") or {}):
        for key in ("command", "aggregation"):
            command = section.get(key)
            if command is not None:
                command.pop("source_sha256", None)
                if command.get("uses_model") is False:
                    command.pop("uses_model")
    return value


def _judge_resources(
    usage: dict[str, int], seconds: float, diagnostics: dict[str, Any] | None = None,
    run_ids: list[str] | None = None,
) -> dict[str, Any]:
    resources: dict[str, Any] = {"usage": usage, "wall_seconds": seconds}
    if diagnostics and "model_usage_complete" in diagnostics:
        resources["usage_complete"] = diagnostics["model_usage_complete"]
        resources["run_id"] = diagnostics.get("model_run_id")
    if run_ids:
        resources["run_ids"] = run_ids
    return resources


def create_app(
    settings: Settings,
    solve: Callable[[Path, str], SolveResult] | None = None,
    complete: ModelClient | Callable[[list[dict[str, str]]], str] | None = None,
) -> FastAPI:
    dataset_bytes = settings.dataset_path.read_bytes()
    examples = [
        Example.model_validate_json(line) for line in dataset_bytes.splitlines() if line.strip()
    ]
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("evaluation dataset must have unique example IDs")
    metric = load_metric(settings.metric_path)
    metric_package = metric_package_provenance(settings.metric_path)
    command_uses_model = any(
        command is not None and command.uses_model
        for command in (metric.command, metric.aggregation)
    )
    if any(not item.id.strip() for item in examples):
        raise ValueError("evaluation examples must not contain blank IDs")
    if metric.kind == "llm":
        if any(not item.reference or not item.reference.strip() for item in examples):
            raise ValueError("LLM evaluation requires nonempty references")
        if not settings.model.strip() or not settings.model_base_url.strip():
            raise ValueError("LLM evaluation requires a model and endpoint")

    def model_client() -> ModelClient:
        return ModelClient(
            settings.model_base_url, settings.model, settings.model_api_key, settings.parameters,
            timeout=settings.model_timeout, retries=MODEL_RETRIES,
            retry_delay=settings.retry_delay,
        )

    json.dumps(settings.parameters, allow_nan=False)
    if metric.kind == "llm" or command_uses_model:
        model_client()
    canonical_examples = sorted(
        ((example.input, example.reference) for example in examples),
        key=lambda item: (item[0], item[1] is not None, item[1] or ""),
    )
    dataset_digest = hashlib.sha256(
        json.dumps(canonical_examples, ensure_ascii=False, separators=(",", ":")).encode(),
    ).hexdigest()
    root = settings.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    candidate_root = root / "candidates"
    usage_dir = (settings.usage_dir or root / "usage").resolve()
    if usage_dir.is_relative_to(candidate_root):
        raise ValueError("usage directory must be outside candidate directories")
    tracking_paths = [
        path.resolve() for path in (
            settings.trace_dir, settings.tracking_dir, settings.tracking_queue_dir,
            settings.slot_dir,
            settings.build_dir,
            settings.search_journal, settings.search_stop_path,
        ) if path is not None
    ]
    if any(path.is_relative_to(candidate_root) for path in tracking_paths):
        raise ValueError("tracking must be outside candidate directories")
    for private_path in (settings.dataset_path, settings.metric_path):
        if private_path.resolve().is_relative_to(candidate_root):
            raise ValueError("private data must be outside candidate directories")
    if settings.assets_dir is not None:
        assets = settings.assets_dir.resolve()
        private_paths = (
            settings.dataset_path.resolve(), settings.metric_path.resolve(), root, usage_dir,
            *tracking_paths,
        )
        if any(path.is_relative_to(assets) for path in private_paths):
            raise ValueError("assets must not contain private data or evaluator state")
        if assets.is_relative_to(root) or assets.is_relative_to(usage_dir):
            raise ValueError("assets must be outside evaluator state and usage directory")
        if any(assets.is_relative_to(path) for path in tracking_paths):
            raise ValueError("assets must be outside tracking directories")
    command = command_provenance(metric)
    if command and any(
        (Path("/") / argument).resolve().is_relative_to(candidate_root)
        for config in command.values() if config
        for argument in config["argv"]
    ):
        raise ValueError("metric implementation must be outside candidate directories")
    model_access = None
    if command_uses_model:
        model_access = MetricModelAccess(
            model=settings.model, base_url=settings.runner_model_base_url, usage_dir=usage_dir,
            model_timeout=settings.model_timeout, retry_delay=settings.retry_delay,
        )
    judge_lock = threading.Lock()

    def make_judge():
        judge_usage = empty_usage()
        if metric.kind != "llm":
            return None, judge_usage
        judge_complete = complete or model_client()
        usage_owner = (
            judge_complete if isinstance(judge_complete, ModelClient)
            else getattr(judge_complete, "__self__", None)
        )
        if isinstance(usage_owner, ModelClient):
            return judge_complete, usage_owner.usage
        callback = cast(Callable[[list[dict[str, str]]], str], judge_complete)

        def tracked_complete(messages: list[dict[str, str]]) -> str:
            add_usage(judge_usage, None)
            return callback(messages)

        return tracked_complete, judge_usage

    harness_runner = None
    if solve is None:
        harness_runner = make_runner(settings, usage_dir)
        if settings.build_dir is not None:
            validate_build_environment(
                settings.build_dir, build_environment(harness_runner, settings.parameters),
                settings.input_assets_digest,
            )
        solve = harness_runner
    directory = root / dataset_digest
    directory.mkdir(mode=0o700, exist_ok=True)
    provenance = {
        "dataset_digest": dataset_digest,
        "metric": metric.model_dump(),
        "metric_command": command,
        "split": settings.split,
        "model": settings.model,
        "model_base_url": settings.model_base_url,
        "parameters": json.loads(json.dumps(settings.parameters, allow_nan=False)),
        "runner_image": settings.runner_image,
        "uses_model": settings.uses_model,
        "harnesses": settings.harnesses,
        "runner": harness_runner.provenance() if harness_runner else None,
        "runner_network": settings.runner_network,
        "runner_model_base_url": settings.runner_model_base_url,
        "runner_timeout": settings.runner_timeout,
        "model_timeout": settings.model_timeout,
        "prepare_timeout": settings.prepare_timeout,
        "model_retries": MODEL_RETRIES,
        "retry_delay": settings.retry_delay,
        "usage_dir": str(usage_dir),
        "ranking_policy": RANKING_POLICY,
        "assets_dir": str(settings.assets_dir.resolve()) if settings.assets_dir else None,
        "finalists": settings.finalists,
        "output_count": settings.output_count,
    }

    if metric_package is not None:
        provenance["metric_package"] = metric_package
    normalized_provenance = _normalized_provenance(provenance)

    def read_record(path: Path) -> dict[str, Any] | None:
        record_path = path / "record.json"
        return json.loads(record_path.read_text()) if record_path.exists() else None

    def execution_lease(path: Path) -> BinaryIO:
        lease = (path / "execution.lock").open("a+b")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lease.close()
            raise
        return lease

    def recover(path: Path) -> None:
        existing = read_record(path)
        if existing:
            if _normalized_provenance(existing["provenance"]) != normalized_provenance:
                raise ValueError("evaluation dataset configuration is frozen")
            if existing["status"] == "running":
                try:
                    lease = execution_lease(path)
                except BlockingIOError:
                    pass
                else:
                    with lease:
                        latest = read_record(path)
                        if latest and latest["status"] == "running":
                            latest["status"] = "failed"
                            latest["failure"] = "interrupted"
                            _write_json(path / "record.json", latest)

    if settings.split == "public":
        directory = directory / "public"
        directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / "reservation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if settings.split == "public":
            config_path = directory / "config.json"
            if config_path.exists() and (
                _normalized_provenance(json.loads(config_path.read_text())) != normalized_provenance
            ):
                raise ValueError("evaluation dataset configuration is frozen")
            if not config_path.exists():
                _write_json(config_path, provenance)
            for path in directory.glob("*/record.json"):
                recover(path.parent)
        else:
            recover(directory)

    def stage_calls(
        name: str, inputs: Any, outputs: Any, started: int, run_ids: list[str],
        *, failed: bool = False,
    ) -> list[dict[str, Any]]:
        if settings.trace_dir is None and settings.tracking_dir is None:
            return []
        span_id = str(uuid4())
        calls = [{
            "name": name, "span_type": "CHAIN", "span_id": span_id,
            "inputs": inputs, "outputs": outputs, "attributes": {"run_ids": run_ids},
            "start_time_ns": started, "end_time_ns": time.time_ns(),
            "status": "ERROR" if failed else "OK",
        }]
        model_calls, errors = collect_model_calls(settings.trace_dir, run_ids, parent_id=span_id)
        calls.extend(model_calls)
        if errors:
            calls[0]["attributes"]["trace_errors"] = errors
        return calls

    def candidate_preparation(candidate_dir: Path, digest: str) -> dict[str, Any] | None:
        if harness_runner is not None:
            return harness_runner.preparation_receipt(candidate_dir, digest)
        if settings.runner_supports_build:
            return load_build_receipt(settings.build_dir, candidate_dir, digest)
        return None

    def measure_candidate(
        candidate: dict[str, Any], details_path: Path, judge_complete, judge_usage,
    ) -> dict[str, Any]:
        details = []
        document: dict[str, Any] = {"candidate_id": candidate["id"], "examples": details}
        preparation = candidate_preparation(Path(candidate["path"]), candidate["digest"])
        preparation_usage = empty_usage()
        preparation_seconds = 0.0
        if preparation is not None:
            document["preparation"] = preparation
            resources = preparation["resources"]["solver"]
            preparation_usage = _usage(resources["usage"])
            preparation_seconds = validate_timing(
                resources["wall_seconds"], "preparation wall time", allow_zero=True,
            )
            _write_json(details_path, document)
        access = replace(model_access, run_ids=[]) if model_access is not None else None
        for example in examples:
            solver_started = time.time_ns()
            try:
                solved = solve(Path(candidate["path"]), example.input)
            except Exception as error:
                failed_usage = None
                failed_seconds = None
                run_id = None
                if isinstance(error, RunnerError):
                    if error.usage is not None:
                        try:
                            failed_usage = _usage(error.usage)
                        except ValueError:
                            pass
                    if isinstance(error.wall_seconds, (int, float)):
                        try:
                            failed_seconds = validate_timing(
                                error.wall_seconds, "solver wall time", allow_zero=True,
                            )
                        except ValueError:
                            pass
                    if isinstance(error.run_id, str) and len(error.run_id) <= 64:
                        run_id = error.run_id
                failed_detail = {
                    "example_id": example.id, "failure": "solver_failure",
                    "error_type": type(error).__name__[:100],
                    "resources_complete": False, "run_id": run_id,
                    "resources": {
                        "solver": {
                            "usage": failed_usage, "wall_seconds": failed_seconds,
                            **({"run_id": run_id} if run_id else {}),
                        },
                        "judge": {"usage": empty_usage(), "wall_seconds": 0.0},
                    },
                }
                calls = stage_calls(
                    "solver", {"example_id": example.id, "input": example.input},
                    {"failure": "solver_failure"}, solver_started,
                    [run_id] if run_id else [], failed=True,
                )
                if calls:
                    failed_detail["calls"] = calls
                details.append(failed_detail)
                _write_json(details_path, document)
                raise
            if not isinstance(solved, SolveResult) or not isinstance(solved.answer, str):
                raise ValueError("solver must return an answer with measured resources")
            solver_usage = _usage(solved.usage)
            if not isinstance(solved.wall_seconds, (int, float)):
                raise ValueError("solver wall time must be numeric")
            solver_seconds = validate_timing(
                solved.wall_seconds, "solver wall time", allow_zero=True,
            )
            detail: dict[str, Any] = {
                "example_id": example.id, "answer": solved.answer, "status": "judging",
                "resources": {
                    "solver": {
                        "usage": solver_usage, "wall_seconds": solver_seconds,
                        **({"run_id": solved.run_id} if solved.run_id else {}),
                    },
                },
            }
            calls = stage_calls(
                "solver", {"example_id": example.id, "input": example.input},
                {"answer": solved.answer}, solver_started,
                [solved.run_id] if solved.run_id else [],
            )
            if calls:
                detail["calls"] = calls
            details.append(detail)
            _write_json(details_path, document)
            with judge_lock:
                before_judge = _usage(judge_usage)
                started = time.monotonic()
                judge_started = time.time_ns()
                before_ids = len(access.run_ids) if access is not None else 0
                result: dict[str, Any] = {"failure": "judge_failure"}
                try:
                    result = evaluate_answer(
                        metric, example, solved.answer, judge_complete,
                        retries=MODEL_RETRIES, retry_delay=settings.retry_delay,
                        model_access=access, usage=judge_usage,
                    )
                except CommandMetricError as error:
                    result["diagnostics"] = error.diagnostics
                    if error.diagnostics.get("model_usage_complete") is False:
                        result["resources_complete"] = False
                    raise
                finally:
                    judge_seconds = validate_timing(
                        time.monotonic() - started, "judge wall time", allow_zero=True,
                    )
                    after_judge = _usage(judge_usage)
                    example_judge_usage = _usage({
                        key: after_judge[key] - before_judge[key] for key in before_judge
                    })
                    run_ids = access.run_ids[before_ids:] if access is not None else []
                    detail.pop("status", None)
                    detail.update(result)
                    detail["resources"]["judge"] = _judge_resources(
                        example_judge_usage, judge_seconds, result.get("diagnostics"), run_ids,
                    )
                    calls = stage_calls(
                        "judge", {"example": example.model_dump(), "answer": solved.answer},
                        result, judge_started, run_ids, failed="failure" in result,
                    )
                    if calls:
                        detail.setdefault("calls", []).extend(calls)
                    _write_json(details_path, document)
        with judge_lock:
            before_aggregation = _usage(judge_usage)
            aggregate_started = time.monotonic()
            aggregation_started = time.time_ns()
            before_ids = len(access.run_ids) if access is not None else 0
            aggregation: dict[str, Any] = {"status": "failed"}
            try:
                candidate_score = aggregate_results(
                    metric, details, model_access=access, usage=judge_usage,
                )
                aggregation.update({"status": "complete", "score": candidate_score})
            except CommandMetricError as error:
                aggregation["diagnostics"] = error.diagnostics
                if error.diagnostics.get("model_usage_complete") is False:
                    aggregation["resources_complete"] = False
                raise
            finally:
                aggregation_seconds = validate_timing(
                    time.monotonic() - aggregate_started, "aggregation time", allow_zero=True,
                )
                after_aggregation = _usage(judge_usage)
                aggregation_usage = _usage({
                    key: after_aggregation[key] - before_aggregation[key]
                    for key in before_aggregation
                })
                aggregation["resources"] = {
                    "judge": _judge_resources(
                        aggregation_usage, aggregation_seconds, aggregation.get("diagnostics"),
                        access.run_ids[before_ids:] if access is not None else [],
                    ),
                }
                calls = stage_calls(
                    "aggregation", {"example_ids": [example.id for example in examples]},
                    {"status": aggregation["status"], "score": aggregation.get("score")},
                    aggregation_started,
                    access.run_ids[before_ids:] if access is not None else [],
                    failed=aggregation["status"] != "complete",
                )
                if calls:
                    aggregation["calls"] = calls
                document["aggregation"] = aggregation
                _write_json(details_path, document)
        return {
            "candidate_id": candidate["id"],
            "digest": candidate["digest"],
            "score": candidate_score,
            **{
                f"{role}_{name}": sum(
                    item["resources"][role]["usage"][key] for item in details
                ) + (aggregation_usage[key] if role == "judge" else preparation_usage[key])
                for role in ("solver", "judge")
                for name, key in (
                    ("tokens", "total_tokens"),
                    ("unreported_attempts", "unreported_attempts"),
                )
            },
            **{
                f"{role}_seconds": validate_timing(
                    sum(item["resources"][role]["wall_seconds"] for item in details)
                    + (aggregation_seconds if role == "judge" else preparation_seconds),
                    f"{role} total time", allow_zero=True,
                )
                for role in ("solver", "judge")
            },
        }

    def candidate_score(
        candidate: dict[str, Any], run_dir: Path, judge_complete, judge_usage,
    ) -> dict[str, Any]:
        details_path = run_dir / "details" / f"{candidate['id']}.json"
        if settings.split == "private":
            return measure_candidate(candidate, details_path, judge_complete, judge_usage)
        cache_dir = directory / "candidate-results" / candidate["digest"]
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cache_path = cache_dir / "result.json"
        cached_details = cache_dir / "details.json"
        with (cache_dir / "execution.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text())
                    if cached["status"] != "complete":
                        raise RuntimeError("public candidate evaluation failed or was interrupted")
                    return {**cached["score"], "candidate_id": candidate["id"]}
                _write_json(cache_path, {"status": "running"})
                try:
                    score = measure_candidate(
                        candidate, cached_details, judge_complete, judge_usage,
                    )
                except Exception:
                    _write_json(cache_path, {"status": "failed"})
                    raise
                _write_json(cache_path, {"status": "complete", "score": score})
                return score
            finally:
                if cached_details.exists():
                    details = json.loads(cached_details.read_text())
                    details["candidate_id"] = candidate["id"]
                    _write_json(details_path, details)

    def export_evaluation(record: dict[str, Any], run_dir: Path) -> None:
        if settings.tracking_dir is None:
            return
        scores = {score["candidate_id"]: score for score in record.get("scores", [])}
        queue = settings.tracking_queue_dir or settings.tracking_dir / "queue"
        for candidate in record["candidates"]:
            details_path = run_dir / "details" / f"{candidate['id']}.json"
            if not details_path.exists():
                continue
            details = json.loads(details_path.read_text())
            score = scores.get(candidate["id"], {})
            finished = bool(score) or details.get("aggregation", {}).get("status") == "complete"
            stages = [
                details.get("preparation", {}), *details["examples"],
                details.get("aggregation", {}),
            ]
            calls = [call for stage in stages for call in stage.get("calls", [])]
            metrics = {key: value for key, value in score.items()
                       if key not in {"candidate_id", "digest"}}
            if not score:
                if finished:
                    metrics["score"] = details["aggregation"]["score"]
                for role in ("solver", "judge"):
                    measured = [
                        stage["resources"][role] for stage in stages
                        if role in stage.get("resources", {})
                    ]
                    usages = [item["usage"] for item in measured if item.get("usage")]
                    seconds = [item["wall_seconds"] for item in measured
                               if item.get("wall_seconds") is not None]
                    if usages:
                        metrics[f"{role}_tokens"] = sum(item["total_tokens"] for item in usages)
                        metrics[f"{role}_unreported_attempts"] = sum(
                            item["unreported_attempts"] for item in usages
                        )
                    if seconds:
                        metrics[f"{role}_seconds"] = sum(seconds)
                    metrics[f"{role}_usage_complete"] = int(
                        len(usages) == len(measured)
                        and all(item["unreported_attempts"] == 0 for item in usages)
                    )
            now = time.time_ns()
            exported = {
                "split": settings.split, "status": "FINISHED" if finished else "FAILED",
                "params": {
                    "campaign_id": settings.release_id, "case_id": settings.case_id,
                    "candidate_id": candidate["id"], "candidate_digest": candidate["digest"],
                    **solver_metadata(settings, harness_runner, Path(candidate["path"])),
                    "metric_version": metric.version,
                    "direction": metric.direction,
                },
                "metrics": metrics, "calls": calls,
                "start_time_ns": min((call["start_time_ns"] for call in calls), default=now),
                "end_time_ns": max((call["end_time_ns"] for call in calls), default=now),
                "examples": details["examples"], "aggregation": details.get("aggregation"),
                **({"preparation": details["preparation"]} if "preparation" in details else {}),
            }
            path = settings.tracking_dir / record["batch_digest"] / f"{candidate['id']}.json"
            _write_json(path, exported)
            receipts = export_configured(path, queue)
            _write_json(path.with_suffix(".tracking.json"), receipts)

    def evaluate(record: dict[str, Any], lease: BinaryIO, run_dir: Path) -> None:
        try:
            judge_complete, judge_usage = make_judge()
            scores = []
            for candidate in record["candidates"]:
                scores.append(candidate_score(candidate, run_dir, judge_complete, judge_usage))
            record["scores"] = scores
            if settings.split == "private":
                record["winners"] = choose_winners(
                    scores, settings.output_count, metric.direction,
                )
            record["status"] = "complete"
        except Exception as error:
            record["status"] = "failed"
            failure: dict[str, Any] = {
                "type": type(error).__name__[:100], "message": str(error)[:2000],
            }
            if isinstance(error, CommandMetricError):
                failure["diagnostics"] = error.diagnostics
                if error.diagnostics.get("model_usage_complete") is False:
                    failure["resources_complete"] = False
            record["failure"] = failure
        finally:
            try:
                _write_json(run_dir / "record.json", record)
                try:
                    export_evaluation(record, run_dir)
                except Exception as error:
                    _write_json(run_dir / "tracking-error.json", {
                        "status": "tracking_pending", "error_type": type(error).__name__,
                    })
            finally:
                lease.close()

    def execute(
        record: dict[str, Any], lease: BinaryIO, run_dir: Path, completion: Future[None],
        started: threading.Event,
        slot: BinaryIO | None,
    ) -> None:
        started.set()
        try:
            try:
                evaluate(record, lease, run_dir)
            finally:
                if slot is not None:
                    slot.close()
        except BaseException as error:
            completion.set_exception(error)
        else:
            completion.set_result(None)

    async def wait_for_evaluation(completion: Future[None]) -> None:
        await asyncio.shield(asyncio.wrap_future(completion))

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        RequestBoundary, token=settings.token,
        max_body_bytes=settings.max_body_bytes or MAX_BODY_BYTES,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_submission(request: Any, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": "invalid submission"}, status_code=422)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/search-readiness")
    def search_readiness() -> dict[str, Any]:
        campaign = settings.search_mode == "campaign"
        enabled = campaign and settings.split == "private"
        if campaign:
            try:
                state = Journal(cast(Path, settings.search_journal)).status()
                if (
                    state["max_solutions"] != settings.search_max_solutions
                    or state["remaining_proposals"] < 0
                ):
                    raise ValueError("search proposal limit differs from configured campaign")
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
                raise HTTPException(503, "search admission unavailable") from None
        return {
            "mode": settings.search_mode, "gate_enabled": enabled,
            "journal_ready": True if campaign else None,
            "max_solutions": settings.search_max_solutions,
            "finalists": settings.finalists, "output_count": settings.output_count,
            "evaluation_enabled": bool(examples), "release_id": settings.release_id,
        }

    @app.get("/v1/evaluations")
    def status() -> dict[str, Any]:
        if settings.split == "public":
            raise HTTPException(422, "evaluation ID is required")
        record = read_record(directory)
        if record is None:
            raise HTTPException(404, {"code": "no_evaluation"})
        return _public(record, settings.split)

    @app.get("/v1/evaluations/{evaluation_id}")
    def public_status(evaluation_id: str) -> dict[str, Any]:
        if settings.split != "public" or len(evaluation_id) != 64 or any(
            character not in "0123456789abcdef" for character in evaluation_id
        ):
            raise HTTPException(404, {"code": "no_evaluation"})
        record = read_record(directory / evaluation_id)
        if record is None:
            raise HTTPException(404, {"code": "no_evaluation"})
        return _public(record, settings.split)

    @app.post("/v1/evaluations")
    def submit(submission: Submission, background: BackgroundTasks) -> JSONResponse:
        if not examples:
            raise HTTPException(422, "evaluation split is empty")
        required_count = settings.finalists if settings.split == "private" else 1
        if len(submission.candidates) != required_count:
            raise HTTPException(422, "invalid submission")
        digest = submission.digest()
        run_dir = directory / digest if settings.split == "public" else directory
        with (directory / "reservation.lock").open("a+b") as lock, ExitStack() as admission_guard:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = read_record(run_dir)
            if existing is not None:
                if existing["batch_digest"] != digest:
                    raise HTTPException(409, "private evaluation is sealed")
                return JSONResponse(_public(existing, settings.split))
            search_admission = None
            if settings.split == "private" and settings.search_journal is not None:
                try:
                    search_admission = admission_guard.enter_context(
                        Journal(settings.search_journal).private_admission(
                            submission, max_solutions=cast(int, settings.search_max_solutions),
                            human_stop_path=settings.search_stop_path,
                        ),
                    )
                except SearchError:
                    raise HTTPException(409, "search is not ready for private evaluation") from None
                except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
                    raise HTTPException(503, "search admission unavailable") from None
            try:
                slot = acquire_slot(settings.slot_dir, settings.execution_slots)
            except CapacityError:
                raise HTTPException(
                    503, "evaluation capacity is busy", headers={"Retry-After": "60"},
                ) from None
            except (OSError, ValueError):
                raise HTTPException(503, "evaluation unavailable") from None
            lease = None
            try:
                run_dir.mkdir(mode=0o700, exist_ok=True)
                lease = execution_lease(run_dir)
                candidates = []
                dataset_candidates = candidate_root / dataset_digest
                dataset_candidates.mkdir(parents=True, exist_ok=True, mode=0o700)
                batch_dir = Path(tempfile.mkdtemp(prefix=digest + "-", dir=dataset_candidates))
                for candidate in submission.candidates:
                    candidate_dir = batch_dir / candidate.id
                    materialize_candidate(candidate, candidate_dir)
                    if harness_runner is not None:
                        try:
                            harness_runner.selection(candidate_dir)
                        except RunnerError:
                            raise HTTPException(422, "invalid submission") from None
                        harness_runner.validate_candidate(candidate_dir)
                    candidate_preparation(candidate_dir, candidate.digest())
                    candidates.append({
                        "id": candidate.id,
                        "digest": candidate.digest(),
                        "path": str(candidate_dir),
                    })
                record = {
                    "status": "running", "batch_digest": digest, "candidates": candidates,
                    "provenance": provenance, "release_id": settings.release_id,
                }
                if search_admission is not None:
                    record["search_admission"] = search_admission
                _write_json(run_dir / "record.json", record)
            except Exception as error:
                if lease is not None:
                    lease.close()
                if slot is not None:
                    slot.close()
                if isinstance(error, HTTPException):
                    raise
                raise HTTPException(503, "evaluation unavailable") from None
            admission_guard.close()
            try:
                response = JSONResponse(_public(record, settings.split), status_code=202)
                completion: Future[None] = Future()
                started = threading.Event()
                worker = threading.Thread(
                    target=execute, args=(record, lease, run_dir, completion, started, slot),
                    daemon=True,
                )
                worker.start()
                started.wait()
            except Exception:
                try:
                    record["status"] = "failed"
                    record["failure"] = "worker_start_failed"
                    _write_json(run_dir / "record.json", record)
                finally:
                    lease.close()
                    if slot is not None:
                        slot.close()
                raise HTTPException(503, "evaluation unavailable") from None
            background.add_task(wait_for_evaluation, completion)
            return response

    return app
