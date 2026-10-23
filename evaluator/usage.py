import fcntl
import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID, uuid4

from evaluator.reliability import add_usage as accumulate_usage
from evaluator.reliability import empty_usage, parse_usage


def _paths(directory: Path, run_id: str) -> tuple[Path, Path]:
    if not isinstance(run_id, str) or str(UUID(run_id)) != run_id:
        raise ValueError("Usage ID must be a canonical UUID")
    return directory / f"{run_id}.json", directory / f"{run_id}.lock"


def _load(path: Path) -> dict[str, int | bool | str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {*empty_usage(), "active_requests", "closed", "capability_digest"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Invalid usage record")
    if any(
        type(value[key]) is not int or value[key] < 0
        for key in fields - {"closed", "capability_digest"}
    ):
        raise ValueError("Invalid usage counters")
    digest = value["capability_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid usage capability digest")
    if type(value["closed"]) is not bool:
        raise ValueError("Invalid usage lifecycle")
    if value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
        raise ValueError("Inconsistent usage total")
    if value["unreported_attempts"] > value["attempts"]:
        raise ValueError("Inconsistent usage attempts")
    return value


def _counters(value: dict[str, int | bool | str]) -> dict[str, int]:
    return {key: int(value[key]) for key in empty_usage()}


def _save(path: Path, value: Mapping[str, int | bool | str]) -> None:
    temporary = path.with_name(f"{path.stem}.{uuid4()}.tmp")
    try:
        temporary.touch(mode=0o600, exist_ok=False)
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_usage(directory: Path, run_id: str) -> str:
    path, lock_path = _paths(directory, run_id)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            raise FileExistsError(path)
        capability = secrets.token_hex(32)
        _save(path, {
            **empty_usage(), "active_requests": 0, "closed": False,
            "capability_digest": hashlib.sha256(capability.encode()).hexdigest(),
        })
        return capability


def read_usage(directory: Path, run_id: str) -> dict[str, int]:
    path, lock_path = _paths(directory, run_id)
    with lock_path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        return _counters(_load(path))


def ensure_open(directory: Path, run_id: str) -> None:
    path, lock_path = _paths(directory, run_id)
    with lock_path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        value = _load(path)
        if value["closed"]:
            raise ValueError("Usage run is closed")


def add_usage(
    directory: Path, run_id: str, response_usage: dict[str, int] | None, *, reserved: bool = False,
) -> None:
    path, lock_path = _paths(directory, run_id)
    usage = None if response_usage is None else parse_usage(response_usage)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = _load(path)
        if value["closed"] and not value["active_requests"]:
            raise ValueError("Usage run is closed")
        counters = _counters(value)
        if reserved:
            if not counters["unreported_attempts"]:
                raise ValueError("Usage run has no reserved attempts")
            counters["attempts"] -= 1
            counters["unreported_attempts"] -= 1
        accumulate_usage(counters, usage)
        value.update(counters)
        _save(path, value)


def begin_request(directory: Path, run_id: str, capability: str) -> None:
    path, lock_path = _paths(directory, run_id)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = _load(path)
        if (
            not isinstance(capability, str)
            or not re.fullmatch(r"[a-f0-9]{64}", capability)
            or not secrets.compare_digest(
                str(value["capability_digest"]), hashlib.sha256(capability.encode()).hexdigest()
            )
        ):
            raise ValueError("Invalid usage capability")
        if value["closed"]:
            raise ValueError("Usage run is closed")
        value["active_requests"] = int(value["active_requests"]) + 1
        _save(path, value)


def end_request(directory: Path, run_id: str) -> None:
    path, lock_path = _paths(directory, run_id)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = _load(path)
        if value["active_requests"] == 0:
            raise ValueError("Usage run has no active requests")
        value["active_requests"] = int(value["active_requests"]) - 1
        _save(path, value)


def seal_usage(directory: Path, run_id: str) -> dict[str, int]:
    path, lock_path = _paths(directory, run_id)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = _load(path)
        value["closed"] = True
        _save(path, value)
        if value["active_requests"]:
            raise ValueError("Usage run has active requests")
        return _counters(value)
