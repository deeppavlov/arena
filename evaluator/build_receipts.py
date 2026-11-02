import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from evaluator.client import ClientError, pack_candidate
from evaluator.reliability import empty_usage


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid build receipt digest")
    return value


def _open(path: Path | str, *, directory: bool, parent: int | None = None) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if directory:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags, dir_fd=parent)
    metadata = os.fstat(descriptor)
    valid_type = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not valid_type or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        os.close(descriptor)
        raise ValueError("Build receipt storage must be operator-owned and not writable by others")
    return descriptor


def _read(parent: int, name: str) -> dict[str, Any]:
    descriptor = _open(name, directory=False, parent=parent)
    with os.fdopen(descriptor, encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("Invalid build receipt document")
    return value


def _configuration(value: dict[str, Any]) -> None:
    if type(value.get("version")) is not int or value["version"] != 2:
        raise ValueError("Unsupported build receipt version")
    for key in ("input_assets_digest", "environment_fingerprint"):
        _digest(value.get(key))
    compatible = value.get("compatible_environment_fingerprints", [])
    if not isinstance(compatible, list):
        raise ValueError("Compatible build environments must be a list")
    for fingerprint in compatible:
        _digest(fingerprint)
    if len(compatible) != len(set(compatible)):
        raise ValueError("Compatible build environments must be unique")


def build_environment(runner: Any, parameters: dict[str, Any]) -> str:
    fields = ("image", "model", "uses_model", "timeout", "model_timeout", "retry_delay",
              "prepare_timeout", "memory", "cpus", "pids_limit", "tmpfs_size", "tmpfs_exec",
              "supports_build", "supports_asset_selection")
    configuration = {
        "parameters": parameters,
        "default": {key: getattr(runner.default_runner, key) for key in fields},
        "registry": {
            name: {key: getattr(item, key) for key in fields}
            for name, item in runner.registry.items()
        },
    }
    encoded = json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_build_environment(
    root: Path | None, environment_fingerprint: str, input_assets_digest: str | None = None,
) -> None:
    if root is None:
        return
    _digest(environment_fingerprint)
    if input_assets_digest is not None:
        _digest(input_assets_digest)
    try:
        root_fd = _open(root, directory=True)
        try:
            configuration = _read(root_fd, "configuration.json")
        finally:
            os.close(root_fd)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read trusted build receipt configuration") from error
    _configuration(configuration)
    if (configuration["environment_fingerprint"] != environment_fingerprint
            and environment_fingerprint not in configuration.get(
                "compatible_environment_fingerprints", [],
            )):
        raise ValueError("Build environment differs from frozen configuration")
    if (input_assets_digest is not None
            and configuration["input_assets_digest"] != input_assets_digest):
        raise ValueError("Build input assets differ from frozen configuration")


def _resources(value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("solver"), dict):
        raise ValueError("Build receipt requires solver resources")
    solver = value["solver"]
    usage = solver.get("usage")
    if (not isinstance(usage, dict) or set(usage) != set(empty_usage())
            or any(type(count) is not int or count < 0 for count in usage.values())):
        raise ValueError("Invalid build receipt usage")
    if (usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
            or usage["unreported_attempts"] > usage["attempts"]
            or usage["attempts"] == 0 and usage["total_tokens"] != 0):
        raise ValueError("Inconsistent build receipt usage")
    seconds = solver.get("wall_seconds")
    try:
        valid_seconds = type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
    except OverflowError:
        valid_seconds = False
    if not valid_seconds or type(solver.get("usage_complete")) is not bool:
        raise ValueError("Invalid build receipt resource measurement")


def read_receipt(
    root: Path, digest: str, require_complete: bool = True,
) -> dict[str, Any]:
    _digest(digest)
    try:
        root_fd = _open(root, directory=True)
        try:
            configuration = _read(root_fd, "configuration.json")
            receipts_fd = _open("receipts", directory=True, parent=root_fd)
            try:
                receipt = _read(receipts_fd, f"{digest}.json")
            finally:
                os.close(receipts_fd)
        finally:
            os.close(root_fd)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read trusted build receipt storage") from error
    _configuration(configuration)
    _configuration(receipt)
    if receipt.get("candidate_digest") != digest:
        raise ValueError("Build receipt candidate digest differs")
    if any(receipt[key] != configuration[key]
           for key in ("input_assets_digest", "environment_fingerprint")):
        raise ValueError("Build receipt differs from frozen configuration")
    status = receipt.get("status")
    if not isinstance(status, str) or status not in {"RUNNING", "FINISHED", "FAILED"}:
        raise ValueError("Unknown build receipt status")
    if require_complete and status != "FINISHED":
        raise ValueError("Build receipt is not FINISHED")
    dependencies = receipt.get("dependency_receipts")
    if not isinstance(dependencies, list):
        raise ValueError("Build receipt requires dependency digests")
    for dependency in dependencies:
        _digest(dependency)
    if len(dependencies) != len(set(dependencies)) or digest in dependencies:
        raise ValueError("Build receipt dependencies must be unique and exclude itself")
    if status == "FINISHED" or "output_digest" in receipt:
        _digest(receipt.get("output_digest"))
    for key in ("resources", "own_resources"):
        if status == "FINISHED" or key in receipt:
            _resources(receipt.get(key))
    return receipt


def load_build_receipt(
    root: Path | None, candidate_dir: Path, candidate_digest: str | None = None,
) -> dict[str, Any] | None:
    build = candidate_dir / "build.py"
    if not build.exists() and not build.is_symlink():
        return None
    if build.is_symlink() or not build.is_file():
        raise ValueError("Candidate build.py must be a regular file")
    if root is None:
        raise ValueError("Candidate requires a trusted build receipt")
    try:
        digest = candidate_digest if candidate_digest is not None else pack_candidate(
            "build", candidate_dir,
        ).digest()
    except ClientError as error:
        raise ValueError("Cannot identify candidate for build receipt") from error
    return read_receipt(root, digest)


def bound_build_receipt(root: Path | None, output_digest: str) -> dict[str, Any] | None:
    if root is None:
        raise ValueError("Asset selection requires a trusted receipt ledger")
    _digest(output_digest)
    try:
        root_fd = _open(root, directory=True)
        try:
            _configuration(_read(root_fd, "configuration.json"))
            bindings_fd = _open("bindings", directory=True, parent=root_fd)
            try:
                try:
                    binding = _read(bindings_fd, f"{output_digest}.json")
                except FileNotFoundError:
                    return None
            finally:
                os.close(bindings_fd)
        finally:
            os.close(root_fd)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read trusted build receipt binding") from error
    receipt = read_receipt(root, _digest(binding.get("candidate_digest")))
    if receipt["output_digest"] != output_digest:
        raise ValueError("Build receipt output differs from its binding")
    return receipt
