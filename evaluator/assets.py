import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

MAX_ASSET_BYTES = 64 * 1024**3
MAX_ASSET_FILES = 100_000
MAX_ASSET_DEPTH = 64
MAX_MANIFEST_BYTES = 64 * 1024**2
MANIFEST_NAME = ".arena-asset.json"
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class AssetError(Exception):
    pass


class AssetLimitError(AssetError):
    pass


def _components(relative: str) -> list[str]:
    parts = relative.split("/")
    if (
        not relative or len(relative) > 4096 or "\\" in relative or "\0" in relative
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise AssetError("asset path must name a relative staging directory")
    if len(parts) > MAX_ASSET_DEPTH:
        raise AssetLimitError("asset directory depth exceeds limit")
    return parts


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _remove_temporary(directory: Path) -> None:
    for parent, directories, _ in os.walk(directory):
        Path(parent).chmod(0o700)
        for name in directories:
            (Path(parent) / name).chmod(0o700)
    shutil.rmtree(directory)


def read_asset_manifest(directory: Path, expected_digest: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise AssetError("invalid asset digest")
    try:
        descriptor = os.open(directory, DIRECTORY_FLAGS)
        try:
            manifest = os.open(
                MANIFEST_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor,
            )
            with os.fdopen(manifest, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise AssetError("asset manifest must be a regular file")
                encoded = stream.read(MAX_MANIFEST_BYTES + 2)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise AssetError("cannot read asset manifest") from error
    if len(encoded) > MAX_MANIFEST_BYTES + 1 or not encoded.endswith(b"\n"):
        raise AssetError("invalid asset manifest encoding")
    encoded = encoded[:-1]
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        raise AssetError("asset manifest digest does not match")
    try:
        result = json.loads(encoded)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise AssetError("invalid asset manifest JSON") from error
    if (
        not isinstance(result, dict) or set(result) != {"version", "entries"}
        or type(result["version"]) is not int or result["version"] != 1
        or not isinstance(result["entries"], list)
        or json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        != encoded
    ):
        raise AssetError("invalid asset manifest schema")
    previous = ""
    directories = {""}
    for entry in result["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise AssetError("invalid asset manifest entry")
        path = entry["path"]
        _components(path)
        parent, _, _ = path.rpartition("/")
        if path <= previous or parent not in directories or path == MANIFEST_NAME:
            raise AssetError("invalid asset manifest paths")
        previous = path
        if entry.get("type") == "directory" and set(entry) == {"path", "type"}:
            directories.add(path)
        elif (
            entry.get("type") == "file"
            and set(entry) == {"path", "type", "size", "sha256", "executable"}
            and type(entry["size"]) is int and entry["size"] >= 0
            and type(entry["executable"]) is bool
            and isinstance(entry["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            continue
        else:
            raise AssetError("invalid asset manifest entry")
    return result


class AssetPublisher:
    def __init__(
        self, staging_root: Path, store_root: Path,
        max_bytes: int = MAX_ASSET_BYTES, max_files: int = MAX_ASSET_FILES,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (max_bytes, max_files)):
            raise ValueError("asset size and entry limits must be positive integers")
        self.staging_root = staging_root
        self.store_root = store_root
        self.max_bytes = max_bytes
        self.max_files = max_files
        staging = staging_root.resolve()
        store = store_root.resolve()
        if staging.is_relative_to(store) or store.is_relative_to(staging):
            raise ValueError("asset staging and trusted store must be separate trees")
        store_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for root in (staging_root, store_root):
            descriptor = os.open(root, DIRECTORY_FLAGS)
            os.close(descriptor)

    def publish(self, relative: str) -> dict[str, object]:
        parts = _components(relative)
        temporary: Path | None = None
        source = -1
        try:
            source = os.open(self.staging_root, DIRECTORY_FLAGS)
            for part in parts:
                child = os.open(part, DIRECTORY_FLAGS, dir_fd=source)
                os.close(source)
                source = child
            with (self.store_root / ".asset-publish.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                temporary = Path(tempfile.mkdtemp(prefix=".asset-", dir=self.store_root))
                entries: list[dict[str, Any]] = []
                totals = {"size": 0, "files": 0, "entries": 0}
                self._copy_tree(source, temporary, "", 0, entries, totals)
                manifest = {"version": 1, "entries": sorted(entries, key=lambda row: row["path"])}
                encoded = json.dumps(
                    manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                ).encode()
                if len(encoded) > MAX_MANIFEST_BYTES:
                    raise AssetLimitError("asset manifest size exceeds limit")
                digest = hashlib.sha256(encoded).hexdigest()
                with (temporary / MANIFEST_NAME).open("xb") as stream:
                    stream.write(encoded + b"\n")
                    stream.flush()
                    os.fchmod(stream.fileno(), 0o444)
                    os.fsync(stream.fileno())
                temporary.chmod(0o555)
                temporary_fd = os.open(temporary, DIRECTORY_FLAGS)
                try:
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                destination = self.store_root / digest
                if destination.exists() or destination.is_symlink():
                    read_asset_manifest(destination, digest)
                    _remove_temporary(temporary)
                else:
                    temporary.rename(destination)
                temporary = None
                store = os.open(self.store_root, DIRECTORY_FLAGS)
                try:
                    os.fsync(store)
                finally:
                    os.close(store)
                return {
                    "digest": digest, "size": totals["size"], "files": totals["files"],
                    "asset_path": f"/assets/derived/{digest}",
                }
        except OSError as error:
            raise AssetError("cannot publish asset directory") from error
        finally:
            if source >= 0:
                os.close(source)
            if temporary is not None:
                _remove_temporary(temporary)

    def _copy_tree(
        self, source: int, destination: Path, relative: str, depth: int,
        entries: list[dict[str, Any]], totals: dict[str, int],
    ) -> None:
        if depth > MAX_ASSET_DEPTH:
            raise AssetLimitError("asset directory depth exceeds limit")
        before = os.fstat(source)
        with os.scandir(source) as children:
            for entry in children:
                totals["entries"] += 1
                if totals["entries"] > self.max_files:
                    raise AssetLimitError("asset entry count exceeds limit")
                if not relative and entry.name == MANIFEST_NAME:
                    raise AssetError("asset contains a reserved manifest path")
                path = f"{relative}/{entry.name}" if relative else entry.name
                _components(path)
                child = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source,
                )
                try:
                    info = os.fstat(child)
                    target = destination / entry.name
                    if stat.S_ISDIR(info.st_mode):
                        target.mkdir(mode=0o700)
                        entries.append({"path": path, "type": "directory"})
                        self._copy_tree(child, target, path, depth + 1, entries, totals)
                    elif stat.S_ISREG(info.st_mode):
                        entries.append(self._copy_file(child, target, path, info, totals))
                    else:
                        raise AssetError("asset must contain only regular files and directories")
                finally:
                    os.close(child)
        if _identity(before) != _identity(os.fstat(source)):
            raise AssetError("asset directory changed during publication")
        target_fd = os.open(destination, DIRECTORY_FLAGS)
        try:
            if relative:
                os.fchmod(target_fd, 0o555)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)

    def _copy_file(
        self, source: int, destination: Path, relative: str,
        before: os.stat_result, totals: dict[str, int],
    ) -> dict[str, Any]:
        digest = hashlib.sha256()
        size = 0
        executable = bool(before.st_mode & 0o111)
        with destination.open("xb") as stream:
            while data := os.read(source, 1024 * 1024):
                size += len(data)
                totals["size"] += len(data)
                if totals["size"] > self.max_bytes:
                    raise AssetLimitError("asset size exceeds limit")
                stream.write(data)
                digest.update(data)
            stream.flush()
            os.fchmod(stream.fileno(), 0o555 if executable else 0o444)
            os.fsync(stream.fileno())
        if size != before.st_size or _identity(before) != _identity(os.fstat(source)):
            raise AssetError("asset file changed during publication")
        totals["files"] += 1
        return {
            "path": relative, "type": "file", "size": size,
            "sha256": digest.hexdigest(), "executable": executable,
        }

class AssetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str


def install_asset_routes(
    app: FastAPI, staging_root: Path, store_root: Path,
    max_bytes: int = MAX_ASSET_BYTES, max_files: int = MAX_ASSET_FILES,
) -> None:
    publisher = AssetPublisher(staging_root, store_root, max_bytes, max_files)

    @app.post("/v1/assets")
    def publish_asset(payload: AssetRequest) -> dict[str, object]:
        try:
            return publisher.publish(payload.path)
        except AssetLimitError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except AssetError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


def publish_remote(
    base_url: str, token_file: Path, path: str, timeout: float,
) -> dict[str, Any]:
    _components(path)
    if not math.isfinite(timeout) or timeout <= 0:
        raise AssetError("asset request timeout must be positive and finite")
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as error:
        raise AssetError("invalid asset service URL") from error
    if (
        url.scheme not in {"http", "https"} or not url.host
        or url.username or url.password or url.query or url.fragment
    ):
        raise AssetError("invalid asset service URL")
    try:
        with token_file.open(encoding="utf-8") as stream:
            token = stream.read(4097).strip()
    except (OSError, UnicodeError) as error:
        raise AssetError("cannot read asset service token file") from error
    if not token or len(token) > 4096 or any(not 33 <= ord(char) <= 126 for char in token):
        raise AssetError("invalid asset service token file")
    endpoint = url.copy_with(path=url.path.rstrip("/") + "/v1/assets")
    try:
        with httpx.Client(follow_redirects=False, trust_env=False, timeout=timeout) as client:
            response = client.request(
                "POST", endpoint, headers={"Authorization": f"Bearer {token}"},
                json={"path": path},
            )
    except httpx.HTTPError as error:
        raise AssetError("asset service request failed") from error
    if not 200 <= response.status_code < 300:
        raise AssetError(f"asset service returned HTTP {response.status_code}")
    try:
        result = response.json()
    except ValueError as error:
        raise AssetError("asset service returned invalid JSON") from error
    if not isinstance(result, dict):
        raise AssetError("asset service returned invalid JSON")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish immutable method assets from staging")
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--url", required=True)
    publish.add_argument("--token-file", type=Path, required=True)
    publish.add_argument("--path", required=True)
    publish.add_argument("--timeout", type=float, default=86400)
    arguments = parser.parse_args(argv)
    try:
        result = publish_remote(
            arguments.url, arguments.token_file, arguments.path, arguments.timeout,
        )
    except AssetError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
