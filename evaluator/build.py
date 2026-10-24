import argparse
import asyncio
import fcntl
import importlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from evaluator.assets import AssetPublisher
from evaluator.build_receipts import bound_build_receipt, read_receipt
from evaluator.candidates import materialize_candidate
from evaluator.capacity import CapacityError, acquire_slot
from evaluator.client import pack_candidate
from evaluator.contracts import Candidate, StrictModel
from evaluator.reliability import empty_usage, validate_timing
from evaluator.runner import HarnessRunner, RunnerError, SolveResult
from evaluator.search import Journal
from evaluator.service import _usage, _write_json


def _save(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    encoded = re.sub(r"(/runs/[a-f0-9-]{36}/)[a-f0-9]{64}(/v1)", r"\1[redacted]\2", encoded)
    _write_json(path, json.loads(encoded))


def run_build(runner: HarnessRunner, candidate_dir: Path, output_dir: Path) -> SolveResult:
    selected, parameters = runner.selection(candidate_dir)
    if not selected.supports_build:
        raise RunnerError("selected harness does not support trusted builds")
    return replace(selected, build_output_dir=output_dir)(candidate_dir, "", parameters=parameters)


class BuildStore:
    def __init__(
        self, root: Path, store_root: Path, input_assets_digest: str, environment_fingerprint: str,
        build: Callable[[Path, Path], SolveResult], *, slot_dir: Path | None = None,
        slots: int = 4, trace_dir: Path | None = None,
        dependency_assets: Callable[[Path], list[str]] | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if any(not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (input_assets_digest, environment_fingerprint)):
            raise ValueError("build input assets and environment require valid identifiers")
        self.root = root
        self.build = build
        self.slot_dir = slot_dir
        self.slots = slots
        self.trace_dir = trace_dir
        self.dependency_assets = dependency_assets
        self.on_complete = on_complete
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("receipts", "jobs", "bindings"):
            (root / name).mkdir(exist_ok=True, mode=0o700)
        self.publisher = AssetPublisher(root / "jobs", store_root)
        self.configuration = {"version": 2, "input_assets_digest": input_assets_digest,
                              "environment_fingerprint": environment_fingerprint}
        with (root / "reservation.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = root / "configuration.json"
            if path.exists() and json.loads(path.read_text()) != self.configuration:
                raise ValueError("build environment is frozen")
            if not path.exists():
                _save(path, self.configuration)
            for path in (root / "receipts").glob("*.json"):
                receipt = read_receipt(root, path.stem, require_complete=False)
                if receipt["status"] != "RUNNING":
                    continue
                with (root / "jobs" / path.stem / "execution.lock").open("a+b") as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    receipt.update(status="FAILED", error="interrupted", end_time_ns=time.time_ns())
                    receipt["own_resources"]["solver"].update(
                        usage_complete=False, wall_seconds_complete=False,
                    )
                    self._total_resources(receipt)
                    _save(path, receipt)
                    if self.on_complete is not None:
                        self.on_complete(receipt)

    def read(self, digest: str) -> dict[str, Any]:
        return read_receipt(self.root, digest, require_complete=False)

    def _dependencies(self, candidate_dir: Path) -> list[str]:
        if self.dependency_assets is None:
            return []
        dependencies: set[str] = set()
        for digest in self.dependency_assets(candidate_dir):
            receipt = bound_build_receipt(self.root, digest)
            if receipt is not None:
                dependencies.add(receipt["candidate_digest"])
                dependencies.update(receipt["dependency_receipts"])
        return sorted(dependencies)

    def _total_resources(self, receipt: dict[str, Any]) -> None:
        stages = [receipt["own_resources"]["solver"]]
        for digest in receipt["dependency_receipts"]:
            stages.append(read_receipt(self.root, digest)["own_resources"]["solver"])
        receipt["resources"] = {"solver": {
            "usage": {key: sum(stage["usage"][key] for stage in stages) for key in empty_usage()},
            "wall_seconds": sum(stage["wall_seconds"] for stage in stages),
            "usage_complete": all(stage.get("usage_complete", True) for stage in stages),
            "wall_seconds_complete": all(stage.get("wall_seconds_complete", True)
                                         for stage in stages),
        }}

    def _bind(self, receipt: dict[str, Any]) -> None:
        binding = self.root / "bindings" / (receipt["output_digest"] + ".json")
        try:
            with binding.open("x", encoding="utf-8") as output:
                os.fchmod(output.fileno(), 0o600)
                json.dump({"candidate_digest": receipt["candidate_digest"]}, output)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError:
            return
        descriptor = os.open(binding.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _evaluate(self, receipt: dict[str, Any], job: Path, started: float) -> None:
        stage = receipt["own_resources"]["solver"]
        try:
            solved = self.build(job / "candidate", job / "output")
            if not isinstance(solved, SolveResult):
                raise ValueError("build runner must return measured resources")
            stage.update(usage=_usage(solved.usage), wall_seconds=validate_timing(
                solved.wall_seconds, "build time", allow_zero=True,
            ), usage_complete=True)
            receipt["run_id"] = solved.run_id
            self._total_resources(receipt)
            _save(self.root / "receipts" / (receipt["candidate_digest"] + ".json"), receipt)
            published = self.publisher.publish(receipt["candidate_digest"] + "/output")
            receipt.update(status="FINISHED", output_digest=published["digest"],
                           output_manifest=published)
        except Exception as error:
            if isinstance(error, RunnerError):
                if error.usage is not None:
                    stage.update(usage=_usage(error.usage), usage_complete=True)
                if error.wall_seconds is not None:
                    stage["wall_seconds"] = validate_timing(
                        error.wall_seconds, "build time", allow_zero=True,
                    )
                receipt["run_id"] = error.run_id
            receipt.update(status="FAILED", error={"type": type(error).__name__,
                                                   "message": str(error)[:2000]})
        finally:
            stage["wall_seconds"] = max(stage["wall_seconds"], time.monotonic() - started)
            receipt["end_time_ns"] = time.time_ns()
            self._total_resources(receipt)
            if receipt.get("run_id") and self.trace_dir is not None:
                try:
                    reader = importlib.import_module("evaluator.trace_store").read_calls
                    receipt["own_calls"] = reader(self.trace_dir, receipt["run_id"])
                    for call in receipt["own_calls"]:
                        call.setdefault("attributes", {}).update(role="solver", phase="preparation")
                except Exception as error:
                    receipt["trace_error"] = type(error).__name__
            receipt["calls"] = list(receipt["own_calls"])
            for digest in receipt["dependency_receipts"]:
                receipt["calls"].extend(read_receipt(self.root, digest).get("own_calls", []))
            with (self.root / "reservation.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if receipt["status"] == "FINISHED":
                    self._bind(receipt)
                _save(self.root / "receipts" / (receipt["candidate_digest"] + ".json"), receipt)
            if self.on_complete is not None:
                self.on_complete(receipt)

    def _execute(
        self, receipt: dict[str, Any], job: Path, lease: BinaryIO, slot: BinaryIO | None,
        started: float, completion: Future[None], launched: threading.Event,
    ) -> None:
        launched.set()
        try:
            self._evaluate(receipt, job, started)
        except BaseException as error:
            completion.set_exception(error)
        else:
            completion.set_result(None)
        finally:
            lease.close()
            if slot is not None:
                slot.close()

    def submit(self, candidate: Candidate) -> tuple[dict[str, Any], Future[None] | None]:
        if "build.py" not in candidate.files:
            raise ValueError("candidate has no build.py")
        digest = candidate.digest()
        path = self.root / "receipts" / (digest + ".json")
        with (self.root / "reservation.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                return self.read(digest), None
            slot = acquire_slot(self.slot_dir, self.slots)
            lease = None
            started = time.monotonic()
            try:
                job = self.root / "jobs" / digest
                with tempfile.TemporaryDirectory(dir=self.root / "jobs") as temporary:
                    staged = Path(temporary)
                    materialize_candidate(candidate, staged / "candidate")
                    dependencies = self._dependencies(staged / "candidate")
                    (staged / "output").mkdir(mode=0o700)
                    staged.rename(job)
                lease = (job / "execution.lock").open("a+b")
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receipt = {
                    **self.configuration, "candidate_digest": digest, "candidate_id": candidate.id,
                    "status": "RUNNING", "dependency_receipts": dependencies,
                    "own_resources": {"solver": {"usage": empty_usage(), "wall_seconds": 0.0,
                                                 "usage_complete": False}},
                    "start_time_ns": time.time_ns(), "calls": [], "own_calls": [],
                }
                self._total_resources(receipt)
                _save(path, receipt)
                completion: Future[None] = Future()
                launched = threading.Event()
                thread = threading.Thread(target=self._execute, args=(
                    receipt, job, lease, slot, started, completion, launched,
                ), daemon=True)
                thread.start()
                launched.wait()
                return json.loads(path.read_text()), completion
            except BaseException:
                if lease is not None:
                    lease.close()
                if slot is not None:
                    slot.close()
                raise


class BuildSubmission(StrictModel):
    candidate: Candidate


def install_build_routes(app: FastAPI, store: BuildStore) -> None:
    async def wait(completion: Future[None]) -> None:
        await asyncio.shield(asyncio.wrap_future(completion))

    @app.post("/v1/builds")
    def submit(payload: BuildSubmission, background: BackgroundTasks) -> JSONResponse:
        try:
            receipt, completion = store.submit(payload.candidate)
        except CapacityError:
            raise HTTPException(503, "evaluation capacity is busy",
                                headers={"Retry-After": "60"}) from None
        except (ValueError, OSError, RunnerError):
            raise HTTPException(422, "invalid build submission") from None
        if completion is not None:
            background.add_task(wait, completion)
        return JSONResponse(receipt, status_code=202 if completion is not None else 200)

    @app.get("/v1/builds/{candidate_digest}")
    def status(candidate_digest: str) -> dict[str, Any]:
        try:
            return store.read(candidate_digest)
        except (ValueError, OSError):
            raise HTTPException(404, "no build receipt") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one registered candidate without task data")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--journal", type=Path, required=True)
    submit.add_argument("--candidate", required=True)
    status = commands.add_parser("status")
    status.add_argument("--digest", required=True)
    for command in (submit, status):
        command.add_argument("--url", required=True)
        command.add_argument("--token-file", type=Path, required=True)
        command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    token = args.token_file.read_text().strip()
    if not token or len(token) > 4096 or any(not 33 <= ord(char) <= 126 for char in token):
        raise ValueError("invalid build submit token")
    url = httpx.URL(args.url)
    if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
            or url.query or url.fragment):
        raise ValueError("invalid build service URL")
    if args.command == "submit":
        journal = Journal(args.journal)
        if journal.status()["stopped"]:
            raise ValueError("campaign stopped; retrieve existing build with status")
        with tempfile.TemporaryDirectory() as directory:
            restored = Path(directory) / "candidate"
            journal.restore(args.candidate, restored)
            candidate = pack_candidate(args.candidate, restored)
        endpoint = url.copy_with(path=url.path.rstrip("/") + "/v1/builds")
        method, payload = "POST", {"candidate": candidate.model_dump()}
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", args.digest):
            raise ValueError("invalid candidate digest")
        endpoint = url.copy_with(path=url.path.rstrip("/") + "/v1/builds/" + args.digest)
        method, payload = "GET", None
    with httpx.Client(timeout=3600, follow_redirects=False, trust_env=False) as client:
        response = client.request(method, endpoint, headers={"Authorization": "Bearer " + token},
                                  json=payload)
    response.raise_for_status()
    receipt = response.json()
    if args.output:
        _save(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
