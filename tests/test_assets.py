import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evaluator.assets import (
    AssetError,
    AssetLimitError,
    AssetPublisher,
    install_asset_routes,
    main,
    read_asset_manifest,
)
from evaluator.service import RequestBoundary


@pytest.fixture
def publisher(tmp_path: Path) -> AssetPublisher:
    staging = tmp_path / "staging"
    staging.mkdir()
    return AssetPublisher(staging, tmp_path / "store")


def test_publish_freezes_bytes_paths_empty_directories_and_executable_mode(
    publisher: AssetPublisher,
) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / "empty").mkdir()
    (source / "weight.bin").write_bytes(b"synthetic weights")
    (source / "search").write_bytes(b"synthetic executable")
    (source / "search").chmod(0o751)
    result = publisher.publish("method")
    destination = publisher.store_root / str(result["digest"])
    assert result["asset_path"] == f"/assets/derived/{result['digest']}"
    assert result["size"] == 37
    assert result["files"] == 2
    assert len(str(result["digest"])) == 64
    assert (destination / "empty").is_dir()
    assert (destination / "search").stat().st_mode & 0o777 == 0o555
    assert (destination / "weight.bin").stat().st_mode & 0o777 == 0o444
    assert destination.stat().st_mode & 0o777 == 0o555
    manifest = json.loads((destination / ".arena-asset.json").read_text())
    assert read_asset_manifest(destination, str(result["digest"])) == manifest
    assert manifest["version"] == 1
    assert [entry["path"] for entry in manifest["entries"]] == ["empty", "search", "weight.bin"]
    assert publisher.publish("method") == result
    (source / "weight.bin").write_bytes(b"modified weights")
    assert (destination / "weight.bin").read_bytes() == b"synthetic weights"
    assert publisher.publish("method")["digest"] != result["digest"]


def test_digest_does_not_depend_on_staging_path_or_nonexecutable_permissions(
    publisher: AssetPublisher,
) -> None:
    for name in ("a", "b"):
        (publisher.staging_root / name).mkdir()
        (publisher.staging_root / name / "data").write_bytes(b"same")
    (publisher.staging_root / "b/data").chmod(0o600)
    assert publisher.publish("a")["digest"] == publisher.publish("b")["digest"]
    (publisher.staging_root / "b/data").chmod(0o700)
    assert publisher.publish("a")["digest"] != publisher.publish("b")["digest"]


@pytest.mark.parametrize("relative", ["", ".", "..", "../outside", "/tmp", "a/../b", "a//b"])
def test_publish_rejects_unsafe_paths(publisher: AssetPublisher, relative: str) -> None:
    with pytest.raises(AssetError):
        publisher.publish(relative)


@pytest.mark.parametrize("kind", ["file", "directory", "selected", "ancestor", "fifo"])
def test_publish_rejects_symlinks_and_special_files(
    publisher: AssetPublisher, tmp_path: Path, kind: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("never copy")
    source = publisher.staging_root / "method"
    source.mkdir()
    relative = "method"
    if kind == "file":
        (source / "secret").symlink_to(outside / "secret")
    elif kind == "directory":
        (source / "link").symlink_to(outside, target_is_directory=True)
    elif kind == "selected":
        (publisher.staging_root / "link").symlink_to(outside, target_is_directory=True)
        relative = "link"
    elif kind == "ancestor":
        (publisher.staging_root / "link").symlink_to(tmp_path, target_is_directory=True)
        relative = "link/outside"
    else:
        os.mkfifo(source / "fifo")
    with pytest.raises(AssetError):
        publisher.publish(relative)
    assert not list(publisher.store_root.glob("[0-9a-f]" * 64))
    assert not list(publisher.store_root.glob(".asset-*/"))


def test_size_and_entry_limits_fail_without_publishing(publisher: AssetPublisher) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / "large").write_bytes(b"0123456789")
    bounded = AssetPublisher(publisher.staging_root, publisher.store_root, max_bytes=9)
    with pytest.raises(AssetLimitError):
        bounded.publish("method")
    (source / "large").unlink()
    (source / "a").mkdir()
    (source / "b").mkdir()
    bounded = AssetPublisher(publisher.staging_root, publisher.store_root, max_files=1)
    with pytest.raises(AssetLimitError):
        bounded.publish("method")
    assert not list(publisher.store_root.glob(".asset-*/"))


def test_manifest_path_is_reserved(publisher: AssetPublisher) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / ".arena-asset.json").write_text("{}")
    with pytest.raises(AssetError):
        publisher.publish("method")


def test_paths_rejected_by_manifest_reader_cannot_be_published(publisher: AssetPublisher) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / "invalid\\name").write_text("data")
    with pytest.raises(AssetError):
        publisher.publish("method")


def test_concurrent_file_mutation_is_rejected(
    publisher: AssetPublisher, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    original = source / "data"
    original.write_bytes(b"original")
    actual_read = os.read
    changed = False

    def read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        data = actual_read(descriptor, size)
        if not changed:
            changed = True
            original.write_bytes(b"changed!")
        return data

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(AssetError, match="changed"):
        publisher.publish("method")


def test_route_requires_existing_boundary_and_rejects_extra_fields(
    publisher: AssetPublisher,
) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / "data").write_text("data")
    app = FastAPI()
    app.add_middleware(RequestBoundary, token="synthetic", max_body_bytes=4096)
    install_asset_routes(app, publisher.staging_root, publisher.store_root)
    client = TestClient(app)
    assert client.post("/v1/assets", json={"path": "method"}).status_code == 401
    headers = {"Authorization": "Bearer synthetic"}
    response = client.post("/v1/assets", json={"path": "method"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["files"] == 1
    assert client.post(
        "/v1/assets", json={"path": "method", "digest": "claimed"}, headers=headers,
    ).status_code == 422
    bad = client.post("/v1/assets", json={"path": "../outside"}, headers=headers)
    assert bad.status_code == 400
    assert "outside" not in bad.text


def test_cli_publishes_only_staging_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("synthetic")

    def request(
        self: httpx.Client, method: str, url: httpx.URL, **kwargs: object,
    ) -> httpx.Response:
        assert method == "POST"
        assert str(url) == "http://train/v1/assets"
        assert kwargs["json"] == {"path": "method"}
        assert kwargs["headers"] == {"Authorization": "Bearer synthetic"}
        return httpx.Response(200, json={"digest": "a" * 64})

    monkeypatch.setattr(httpx.Client, "request", request)
    assert main([
        "publish", "--url", "http://train", "--token-file", str(token_file),
        "--path", "method",
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {"digest": "a" * 64}


def test_concurrent_publication_is_idempotent(publisher: AssetPublisher) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / "data").write_bytes(b"synthetic")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(publisher.publish, ["method"] * 4))
    assert all(result == results[0] for result in results)
    assert not list(publisher.store_root.glob(".asset-*/"))


def test_directory_depth_is_bounded(publisher: AssetPublisher) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    for _ in range(65):
        source /= "d"
        source.mkdir()
    with pytest.raises(AssetLimitError, match="depth"):
        publisher.publish("method")
    assert not list(publisher.store_root.glob(".asset-*/"))


def test_renamed_source_directory_cannot_follow_replacement_symlink(
    publisher: AssetPublisher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = publisher.staging_root / "method"
    source.mkdir()
    (source / "data").write_text("allowed")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data").write_text("secret")
    actual_open = os.open
    replaced = False

    def open_path(path: str | Path, flags: int, mode: int = 0o777, **kwargs: int) -> int:
        nonlocal replaced
        if path == "data" and not replaced:
            replaced = True
            source.rename(publisher.staging_root / "old")
            source.symlink_to(outside, target_is_directory=True)
        return actual_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", open_path)
    with pytest.raises(AssetError, match="changed"):
        publisher.publish("method")
    assert not list(publisher.store_root.glob(".asset-*/"))


@pytest.mark.parametrize("kind", ["digest", "schema", "symlink", "fifo"])
def test_read_manifest_rejects_corruption(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "asset"
    directory.mkdir()
    payload = {"version": 1, "entries": []}
    if kind == "schema":
        payload["version"] = True
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    manifest = directory / ".arena-asset.json"
    if kind == "symlink":
        external = tmp_path / "manifest"
        external.write_bytes(encoded + b"\n")
        manifest.symlink_to(external)
    elif kind == "fifo":
        os.mkfifo(manifest)
    else:
        manifest.write_bytes(encoded + b"\n")
    if kind == "digest":
        digest = "a" * 64
    with pytest.raises(AssetError):
        read_asset_manifest(directory, digest)
