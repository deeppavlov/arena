import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from evaluator.candidates import materialize_candidate
from evaluator.contracts import Example, Metric, Submission
from evaluator.judge import ModelClient
from evaluator.metrics import (
    CommandMetricError,
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
from evaluator.runner import DockerRunner, HarnessRunner, RunnerError, SolveResult

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

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.token, self.runner_image, self.release_id)):
            raise ValueError("token, runner image and release ID are required")
        if type(self.finalists) is not int or self.finalists < 1:
            raise ValueError("finalists must be a positive integer")
        if type(self.output_count) is not int or not 1 <= self.output_count <= self.finalists:
            raise ValueError("output_count must be between 1 and finalists")
        if self.split not in {"private", "public"}:
            raise ValueError("split must be private or public")
        if type(self.uses_model) is not bool:
            raise ValueError("uses_model must be a boolean")
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
                or set(config) - {"image", "uses_model", "network"}
                or not isinstance(config.get("image"), str) or not config["image"].strip()
                or type(config.get("uses_model", self.uses_model)) is not bool
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
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a JSON object")

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
        uses_model = value("USES_MODEL", "true").lower()
        if uses_model not in {"true", "false"}:
            raise ValueError("ARENA_USES_MODEL must be true or false")
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


def create_app(
    settings: Settings,
    solve: Callable[[Path, str], SolveResult] | None = None,
    complete: ModelClient | Callable[[list[dict[str, str]]], str] | None = None,
) -> FastAPI:
    dataset_bytes = settings.dataset_path.read_bytes()
    examples = [
        Example.model_validate_json(line) for line in dataset_bytes.splitlines() if line.strip()
    ]
    if not examples or len({example.id for example in examples}) != len(examples):
        raise ValueError("evaluation dataset must be nonempty with unique example IDs")
    metric = Metric.model_validate_json(settings.metric_path.read_bytes())
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
    if metric.kind == "llm":
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
    for private_path in (settings.dataset_path, settings.metric_path):
        if private_path.resolve().is_relative_to(candidate_root):
            raise ValueError("private data must be outside candidate directories")
    if settings.assets_dir is not None:
        assets = settings.assets_dir.resolve()
        private_paths = (
            settings.dataset_path.resolve(), settings.metric_path.resolve(), root, usage_dir,
        )
        if any(path.is_relative_to(assets) for path in private_paths):
            raise ValueError("assets must not contain private data or evaluator state")
        if assets.is_relative_to(root) or assets.is_relative_to(usage_dir):
            raise ValueError("assets must be outside evaluator state and usage directory")
    command = command_provenance(metric)
    if command and any(
        Path(path).resolve().is_relative_to(candidate_root)
        for config in command.values() if config
        for path in config["source_sha256"]
    ):
        raise ValueError("metric implementation must be outside candidate directories")
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

    def make_runner(config: dict[str, Any]) -> DockerRunner:
        return DockerRunner(
            image=config.get("image", settings.runner_image),
            network=config.get("network", settings.runner_network),
            uses_model=config.get("uses_model", settings.uses_model),
            model=settings.model,
            model_base_url=settings.runner_model_base_url,
            assets_dir=settings.assets_dir,
            timeout=settings.runner_timeout,
            model_timeout=settings.model_timeout,
            prepare_timeout=settings.prepare_timeout,
            retry_delay=settings.retry_delay,
            usage_dir=usage_dir,
        )

    harness_runner = None
    if solve is None:
        harness_runner = HarnessRunner(
            make_runner({}),
            {name: make_runner(config) for name, config in settings.harnesses.items()},
        )
        solve = harness_runner
    directory = root / dataset_digest
    directory.mkdir(mode=0o700, exist_ok=True)
    provenance = {
        "dataset_digest": dataset_digest,
        "dataset_records_digest": hashlib.sha256(
            json.dumps(
                [example.model_dump() for example in examples],
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode(),
        ).hexdigest(),
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
            if existing["provenance"] != provenance:
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
            if config_path.exists() and json.loads(config_path.read_text()) != provenance:
                raise ValueError("evaluation dataset configuration is frozen")
            if not config_path.exists():
                _write_json(config_path, provenance)
            for path in directory.glob("*/record.json"):
                recover(path.parent)
        else:
            recover(directory)

    def measure_candidate(
        candidate: dict[str, Any], details_path: Path, judge_complete, judge_usage,
    ) -> dict[str, Any]:
        details = []
        for example in examples:
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
                details.append({
                    "example_id": example.id, "failure": "solver_failure",
                    "error_type": type(error).__name__[:100],
                    "resources_complete": False, "run_id": run_id,
                    "resources": {
                        "solver": {"usage": failed_usage, "wall_seconds": failed_seconds},
                        "judge": {"usage": empty_usage(), "wall_seconds": 0.0},
                    },
                })
                _write_json(
                    details_path,
                    {"candidate_id": candidate["id"], "examples": details},
                )
                raise
            if not isinstance(solved, SolveResult) or not isinstance(solved.answer, str):
                raise ValueError("solver must return an answer with measured resources")
            solver_usage = _usage(solved.usage)
            if not isinstance(solved.wall_seconds, (int, float)):
                raise ValueError("solver wall time must be numeric")
            solver_seconds = validate_timing(
                solved.wall_seconds, "solver wall time", allow_zero=True,
            )
            with judge_lock:
                before_judge = _usage(judge_usage)
                started = time.monotonic()
                result: dict[str, Any] = {"failure": "judge_failure"}
                try:
                    result = evaluate_answer(
                        metric, example, solved.answer, judge_complete,
                        retries=MODEL_RETRIES, retry_delay=settings.retry_delay,
                    )
                except CommandMetricError as error:
                    result["diagnostics"] = error.diagnostics
                    raise
                finally:
                    judge_seconds = validate_timing(
                        time.monotonic() - started, "judge wall time", allow_zero=True,
                    )
                    after_judge = _usage(judge_usage)
                    example_judge_usage = _usage({
                        key: after_judge[key] - before_judge[key] for key in before_judge
                    })
                    details.append({
                        "example_id": example.id, "answer": solved.answer, **result,
                        "resources": {
                            "solver": {
                                "usage": solver_usage, "wall_seconds": solver_seconds,
                            },
                            "judge": {
                                "usage": example_judge_usage, "wall_seconds": judge_seconds,
                            },
                        },
                    })
                    _write_json(
                        details_path,
                        {"candidate_id": candidate["id"], "examples": details},
                    )
        aggregate_started = time.monotonic()
        aggregation: dict[str, Any] = {"status": "failed"}
        try:
            candidate_score = aggregate_results(metric, details)
            aggregation.update({"status": "complete", "score": candidate_score})
        except CommandMetricError as error:
            aggregation["diagnostics"] = error.diagnostics
            raise
        finally:
            aggregation_seconds = validate_timing(
                time.monotonic() - aggregate_started, "aggregation time", allow_zero=True,
            )
            aggregation["resources"] = {
                "judge": {"usage": empty_usage(), "wall_seconds": aggregation_seconds},
            }
            _write_json(
                details_path,
                {"candidate_id": candidate["id"], "examples": details,
                 "aggregation": aggregation},
            )
        return {
            "candidate_id": candidate["id"],
            "digest": candidate["digest"],
            "score": candidate_score,
            **{
                f"{role}_{name}": sum(
                    item["resources"][role]["usage"][key] for item in details
                )
                for role in ("solver", "judge")
                for name, key in (
                    ("tokens", "total_tokens"),
                    ("unreported_attempts", "unreported_attempts"),
                )
            },
            **{
                f"{role}_seconds": validate_timing(
                    sum(item["resources"][role]["wall_seconds"] for item in details)
                    + (aggregation_seconds if role == "judge" else 0.0),
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

    def evaluate(record: dict[str, Any], lease: BinaryIO, run_dir: Path) -> None:
        try:
            judge_complete, judge_usage = make_judge()
            scores = [
                candidate_score(candidate, run_dir, judge_complete, judge_usage)
                for candidate in record["candidates"]
            ]
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
            record["failure"] = failure
        finally:
            try:
                _write_json(run_dir / "record.json", record)
            finally:
                lease.close()

    def execute(
        record: dict[str, Any], lease: BinaryIO, run_dir: Path, completion: Future[None],
        started: threading.Event,
    ) -> None:
        started.set()
        try:
            evaluate(record, lease, run_dir)
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

    @app.get("/v1/evaluations")
    def status() -> dict[str, Any]:
        if settings.split == "public":
            raise HTTPException(422, "evaluation ID is required")
        record = read_record(directory)
        if record is None:
            raise HTTPException(404, "no evaluation")
        return _public(record, settings.split)

    @app.get("/v1/evaluations/{evaluation_id}")
    def public_status(evaluation_id: str) -> dict[str, Any]:
        if settings.split != "public" or len(evaluation_id) != 64 or any(
            character not in "0123456789abcdef" for character in evaluation_id
        ):
            raise HTTPException(404, "no evaluation")
        record = read_record(directory / evaluation_id)
        if record is None:
            raise HTTPException(404, "no evaluation")
        return _public(record, settings.split)

    @app.post("/v1/evaluations")
    def submit(submission: Submission, background: BackgroundTasks) -> JSONResponse:
        required_count = settings.finalists if settings.split == "private" else 1
        if len(submission.candidates) != required_count:
            raise HTTPException(422, "invalid submission")
        digest = submission.digest()
        run_dir = directory / digest if settings.split == "public" else directory
        with (directory / "reservation.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = read_record(run_dir)
            if existing is not None:
                if existing["batch_digest"] != digest:
                    raise HTTPException(409, "private evaluation is sealed")
                return JSONResponse(_public(existing, settings.split))
            run_dir.mkdir(mode=0o700, exist_ok=True)
            lease = execution_lease(run_dir)
            try:
                candidates = []
                dataset_candidates = candidate_root / dataset_digest
                dataset_candidates.mkdir(parents=True, exist_ok=True, mode=0o700)
                batch_dir = Path(tempfile.mkdtemp(prefix=digest + "-", dir=dataset_candidates))
                for candidate in submission.candidates:
                    candidate_dir = batch_dir / candidate.id
                    materialize_candidate(candidate, candidate_dir)
                    candidates.append({
                        "id": candidate.id,
                        "digest": candidate.digest(),
                        "path": str(candidate_dir),
                    })
                record = {
                    "status": "running", "batch_digest": digest, "candidates": candidates,
                    "provenance": provenance, "release_id": settings.release_id,
                }
                _write_json(run_dir / "record.json", record)
            except Exception:
                lease.close()
                raise HTTPException(503, "evaluation unavailable") from None
            response = JSONResponse(_public(record, settings.split), status_code=202)
            completion: Future[None] = Future()
            started = threading.Event()
            worker = threading.Thread(
                target=execute, args=(record, lease, run_dir, completion, started), daemon=True,
            )
            try:
                worker.start()
                started.wait()
            except Exception:
                try:
                    record["status"] = "failed"
                    record["failure"] = "worker_start_failed"
                    _write_json(run_dir / "record.json", record)
                finally:
                    lease.close()
                raise HTTPException(503, "evaluation unavailable") from None
            background.add_task(wait_for_evaluation, completion)
            return response

    return app
