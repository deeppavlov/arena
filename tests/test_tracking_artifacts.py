import json
import shutil
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from evaluator.tracking_artifacts import log_json_artifact, read_json_artifact


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.uploads: list[tuple[str, int]] = []
        self.dictionaries: list[str] = []
        self.fail_once: str | None = None

    def log_dict(self, value: Any, artifact_path: str) -> None:
        self.dictionaries.append(artifact_path)
        self.save(artifact_path, json.dumps(value, indent=2, default=str).encode())

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        path = Path(local_path)
        target = f"{artifact_path}/{path.name}" if artifact_path else path.name
        self.save(target, path.read_bytes())

    def save(self, target: str, value: bytes) -> None:
        path = self.root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        self.uploads.append((target, len(value)))
        if self.fail_once == target:
            self.fail_once = None
            raise ConnectionError("response lost after storage")

    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str) -> str:
        assert run_id == "run-one"
        source = self.root / artifact_path
        target = Path(dst_path) / artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return str(target)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "server")


def test_small_artifact_preserves_existing_sdk_format(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TMPDIR", raising=False)
    value = {"text": "Привет\n\"☃", "nested": [1, False, None, {"x": "y"}]}
    log_json_artifact(store, store, "run-one", value, "record.json")
    assert store.dictionaries == ["record.json"]
    assert read_json_artifact(store, "run-one", "record.json") == value
    assert store.uploads == [("record.json", len(json.dumps(value, indent=2).encode()))]


@pytest.mark.parametrize("operation", ["log", "read"])
def test_missing_configured_tmpdir_fails_without_fallback(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    store.save("record.json", b"{}")
    store.uploads.clear()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "missing-disk-directory"))
    with pytest.raises(FileNotFoundError):
        if operation == "log":
            log_json_artifact(store, store, "run-one", {}, "record.json")
        else:
            read_json_artifact(store, "run-one", "record.json")
    assert not store.uploads


def test_large_unicode_artifact_uses_bounded_parts_and_exact_sdk_bytes(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evaluator.tracking_artifacts.MAX_ARTIFACT_BYTES", 512)
    value = {"text": "😀\"\\\n\tПривет" * 600, "date": datetime(2026, 10, 29),
             "nested": [True, None, {4: "four"}], "empty": {}}
    encoded = json.dumps(value, indent=2, default=str).encode()
    log_json_artifact(store, store, "run-one", value, "calls/one.json")
    assert all(size <= 512 for _, size in store.uploads)
    assert store.uploads[-1][0] == "calls/one.json"
    manifest = json.loads((store.root / "calls/one.json").read_bytes())["arena_json_parts"]
    parts = sorted((store.root / manifest["part_prefix"]).glob("*.bin"))
    assert b"".join(path.read_bytes() for path in parts) == encoded
    assert read_json_artifact(store, "run-one", "calls/one.json") == json.loads(encoded)


def test_reserved_manifest_shaped_user_value_round_trips(store: Store) -> None:
    value = {"arena_json_parts": {"version": 37, "arbitrary": "ordinary input"}}
    log_json_artifact(store, store, "run-one", value, "record.json")
    assert read_json_artifact(store, "run-one", "record.json") == value


def test_export_requires_exact_read_back_before_publishing_manifest(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evaluator.tracking_artifacts.MAX_ARTIFACT_BYTES", 512)
    original = store.download_artifacts

    def corrupted(run_id: str, artifact_path: str, dst_path: str) -> str:
        result = original(run_id, artifact_path, dst_path)
        path = Path(result)
        path.write_bytes(b"!" + path.read_bytes()[1:])
        return result

    monkeypatch.setattr(store, "download_artifacts", corrupted)
    with pytest.raises(ValueError, match="read-back differs"):
        log_json_artifact(store, store, "run-one", {"text": "x" * 1200}, "record.json")
    assert not (store.root / "record.json").exists()


@pytest.mark.parametrize("failure", ["record.json.parts/00000001.bin", "record.json"])
def test_lost_ack_reuses_paths_and_preserves_complete_content(
    store: Store, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    monkeypatch.setattr("evaluator.tracking_artifacts.MAX_ARTIFACT_BYTES", 512)
    value = {"text": "1234567890" * 300}
    store.fail_once = failure
    with pytest.raises(ConnectionError):
        log_json_artifact(store, store, "run-one", value, "record.json")
    log_json_artifact(store, store, "run-one", value, "record.json")
    assert read_json_artifact(store, "run-one", "record.json") == value
    first_names = {name for name, _ in store.uploads}
    log_json_artifact(store, store, "run-one", value, "record.json")
    assert {name for name, _ in store.uploads} == first_names


@pytest.mark.parametrize("change", ["missing", "truncated", "extra"])
def test_reader_rejects_incomplete_or_wrong_size_parts(
    store: Store, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    monkeypatch.setattr("evaluator.tracking_artifacts.MAX_ARTIFACT_BYTES", 512)
    log_json_artifact(store, store, "run-one", {"text": "x" * 1200}, "record.json")
    part = store.root / "record.json.parts/00000001.bin"
    if change == "missing":
        part.unlink()
    elif change == "truncated":
        part.write_bytes(part.read_bytes()[:-1])
    else:
        part.write_bytes(part.read_bytes() + b"x")
    with pytest.raises((ValueError, FileNotFoundError)):
        read_json_artifact(store, "run-one", "record.json")


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("part_prefix", "/outside"), ("part_prefix", "../outside"),
    ("part_prefix", "other.parts"), ("part_prefix", "record.json.parts\\other"),
    ("part_count", True), ("part_count", 0), ("part_count", 999),
    ("part_bytes", 0), ("part_bytes", 513), ("total_bytes", -1), ("total_bytes", True),
])
def test_reader_validates_manifest(
    store: Store, monkeypatch: pytest.MonkeyPatch, key: str, value: Any,
) -> None:
    monkeypatch.setattr("evaluator.tracking_artifacts.MAX_ARTIFACT_BYTES", 512)
    log_json_artifact(store, store, "run-one", {"text": "x" * 1200}, "record.json")
    path = store.root / "record.json"
    manifest = json.loads(path.read_bytes())
    manifest["arena_json_parts"][key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_json_artifact(store, "run-one", "record.json")


@pytest.mark.parametrize("path", ["/record.json", "../record.json", "a/../record.json",
                                   "a\\record.json", "./record.json", "record.yaml"])
def test_rejects_unsafe_or_non_json_artifact_paths(store: Store, path: str) -> None:
    with pytest.raises(ValueError):
        log_json_artifact(store, store, "run-one", {}, path)
    with pytest.raises(ValueError):
        read_json_artifact(store, "run-one", path)
    assert not store.uploads


def test_serialization_memory_is_bounded_for_one_large_unicode_string(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evaluator.tracking_artifacts.MAX_ARTIFACT_BYTES", 64 * 1024)
    value = {"text": "😀" * (2 * 1024 * 1024)}
    tracemalloc.start()
    try:
        log_json_artifact(store, store, "run-one", value, "record.json")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024
    assert read_json_artifact(store, "run-one", "record.json") == value
