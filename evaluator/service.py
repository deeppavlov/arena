import base64
import fcntl
import hashlib
import json
import math
import os
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from evaluator.contracts import Example, Metric, Submission
from evaluator.judge import ModelClient, judge_answer
from evaluator.runner import DockerRunner

MAX_BODY_BYTES = 70 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    metric_path: Path
    state_dir: Path
    token: str = field(repr=False)
    model: str
    model_base_url: str
    runner_image: str
    release_id: str
    model_api_key: str = field(default="", repr=False)
    runner_network: str = "none"
    runner_model_base_url: str = "http://model-gateway:8081/v1"
    runner_timeout: float = 300
    assets_dir: Path | None = None
    finalists: int = 5
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (
            self.token, self.model, self.model_base_url, self.runner_image, self.release_id,
        )):
            raise ValueError("token, model, endpoint, runner image and release ID are required")
        if not 1 <= self.finalists <= 5:
            raise ValueError("finalists must be between 1 and 5")
        if not math.isfinite(self.runner_timeout) or self.runner_timeout <= 0:
            raise ValueError("runner timeout must be positive and finite")
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
        return cls(
            dataset_path=Path(value("DATASET_PATH")),
            metric_path=Path(value("METRIC_PATH")),
            state_dir=Path(value("STATE_DIR")),
            token=value("TOKEN"),
            model=value("MODEL"),
            model_base_url=value("MODEL_BASE_URL"),
            runner_image=value("RUNNER_IMAGE"),
            release_id=value("RELEASE_ID"),
            model_api_key=value("MODEL_API_KEY", ""),
            runner_network=value("RUNNER_NETWORK", "none"),
            runner_model_base_url=value("RUNNER_MODEL_BASE_URL", "http://model-gateway:8081/v1"),
            runner_timeout=float(value("RUNNER_TIMEOUT", "300")),
            assets_dir=Path(assets) if assets else None,
            finalists=int(value("FINALISTS", "5")),
            parameters=json.loads(value("PARAMETERS", "{}")),
        )


class RequestBoundary:
    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.authorization = b"Bearer " + token.encode()

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
            length = MAX_BODY_BYTES + 1
        if length < 0 or length > MAX_BODY_BYTES:
            await JSONResponse({"detail": "body too large"}, status_code=413)(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
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


def _public(record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] != "complete":
        return {"status": record["status"]}
    return {"status": "complete", "scores": record["scores"], "winner": record["winner"]}


def create_app(
    settings: Settings,
    solve: Callable[[Path, str], str] | None = None,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> FastAPI:
    dataset_bytes = settings.dataset_path.read_bytes()
    examples = [
        Example.model_validate_json(line) for line in dataset_bytes.splitlines() if line.strip()
    ]
    if not examples or len({example.id for example in examples}) != len(examples):
        raise ValueError("private dataset must be nonempty with unique example IDs")
    if any(
        not value.strip() for item in examples for value in (item.id, item.input, item.reference)
    ):
        raise ValueError("private examples must not contain blank values")
    metric = Metric.model_validate_json(settings.metric_path.read_bytes())
    canonical_examples = sorted((example.input, example.reference) for example in examples)
    dataset_digest = hashlib.sha256(
        json.dumps(canonical_examples, ensure_ascii=False, separators=(",", ":")).encode(),
    ).hexdigest()
    root = settings.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    candidate_root = root / "candidates"
    for private_path in (settings.dataset_path, settings.metric_path):
        if private_path.resolve().is_relative_to(candidate_root):
            raise ValueError("private data must be outside candidate directories")
    if settings.assets_dir is not None:
        assets = settings.assets_dir.resolve()
        private_paths = (settings.dataset_path.resolve(), settings.metric_path.resolve(), root)
        if any(path.is_relative_to(assets) for path in private_paths):
            raise ValueError("assets must not contain private data or evaluator state")
        if assets.is_relative_to(root):
            raise ValueError("assets must be outside evaluator state")
    model_client = ModelClient(
        settings.model_base_url, settings.model, settings.model_api_key, settings.parameters,
    )
    complete = complete or model_client.complete
    solve = solve or DockerRunner(
        image=settings.runner_image,
        network=settings.runner_network,
        model=settings.model,
        model_base_url=settings.runner_model_base_url,
        assets_dir=settings.assets_dir,
        timeout=settings.runner_timeout,
    )
    directory = root / dataset_digest
    directory.mkdir(mode=0o700, exist_ok=True)
    record_path = directory / "record.json"
    provenance = {
        "dataset_digest": dataset_digest,
        "metric": metric.model_dump(),
        "model": settings.model,
        "model_base_url": settings.model_base_url,
        "parameters": json.loads(json.dumps(settings.parameters, allow_nan=False)),
        "runner_image": settings.runner_image,
        "runner_network": settings.runner_network,
        "runner_model_base_url": settings.runner_model_base_url,
        "runner_timeout": settings.runner_timeout,
        "assets_dir": str(settings.assets_dir.resolve()) if settings.assets_dir else None,
        "finalists": settings.finalists,
    }

    def read_record() -> dict[str, Any] | None:
        return json.loads(record_path.read_text()) if record_path.exists() else None

    def execution_lease() -> BinaryIO:
        lease = (directory / "execution.lock").open("a+b")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lease.close()
            raise
        return lease

    with (directory / "reservation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_record()
        if existing:
            if existing["provenance"] != provenance:
                raise ValueError("private dataset configuration is frozen")
            if existing["status"] == "running":
                try:
                    lease = execution_lease()
                except BlockingIOError:
                    pass
                else:
                    with lease:
                        existing["status"] = "failed"
                        existing["failure"] = "interrupted"
                        _write_json(record_path, existing)

    def evaluate(record: dict[str, Any], lease: BinaryIO) -> None:
        try:
            scores = []
            for candidate in record["candidates"]:
                details = []
                for example in examples:
                    answer = solve(Path(candidate["path"]), example.input)
                    if answer.strip():
                        result = judge_answer(metric, example, answer, complete)
                    else:
                        result = {"score": 0.0, "criteria": {}, "failure": "candidate_failure"}
                    details.append({"example_id": example.id, "answer": answer, **result})
                    _write_json(
                        directory / "details" / f"{candidate['id']}.json", {"examples": details},
                    )
                scores.append({
                    "candidate_id": candidate["id"],
                    "digest": candidate["digest"],
                    "score": sum(item["score"] for item in details) / len(details),
                })
            record["scores"] = scores
            best_score = max(item["score"] for item in scores)
            winners = [item["candidate_id"] for item in scores if item["score"] == best_score]
            record["winner"] = winners[0] if len(winners) == 1 else None
            record["status"] = "complete"
        except Exception as error:
            record["status"] = "failed"
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            try:
                _write_json(record_path, record)
            finally:
                lease.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestBoundary, token=settings.token)

    @app.exception_handler(RequestValidationError)
    async def invalid_submission(request: Any, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": "invalid submission"}, status_code=422)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/evaluations")
    def status() -> dict[str, Any]:
        record = read_record()
        if record is None:
            raise HTTPException(404, "no evaluation")
        return _public(record)

    @app.post("/v1/evaluations")
    def submit(submission: Submission, background: BackgroundTasks) -> JSONResponse:
        if len(submission.candidates) != settings.finalists:
            raise HTTPException(422, "invalid submission")
        digest = submission.digest()
        with (directory / "reservation.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = read_record()
            if existing is not None:
                if existing["batch_digest"] != digest:
                    raise HTTPException(409, "private evaluation is sealed")
                return JSONResponse(_public(existing))
            lease = execution_lease()
            try:
                candidates = []
                dataset_candidates = candidate_root / dataset_digest
                dataset_candidates.mkdir(parents=True, exist_ok=True, mode=0o700)
                batch_dir = Path(tempfile.mkdtemp(prefix=digest + "-", dir=dataset_candidates))
                for candidate in submission.candidates:
                    candidate_dir = batch_dir / candidate.id
                    candidate_dir.mkdir(mode=0o755)
                    candidate_dir.chmod(0o755)
                    for name, value in candidate.files.items():
                        path = candidate_dir / name
                        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                        for parent in path.parents:
                            if parent == candidate_dir:
                                break
                            parent.chmod(0o755)
                        path.write_bytes(base64.b64decode(value, validate=True))
                        path.chmod(0o644)
                    candidates.append({
                        "id": candidate.id,
                        "digest": candidate.digest(),
                        "path": str(candidate_dir),
                    })
                record = {
                    "status": "running", "batch_digest": digest, "candidates": candidates,
                    "provenance": provenance, "release_id": settings.release_id,
                }
                _write_json(record_path, record)
            except Exception:
                lease.close()
                raise HTTPException(503, "evaluation unavailable") from None
            background.add_task(evaluate, record, lease)
            return JSONResponse({"status": "running"}, status_code=202)

    return app
