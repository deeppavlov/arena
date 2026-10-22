import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.runner import DockerRunner, HarnessRunner, RunnerError

IMAGE = "sha256:" + "a" * 64


def publish_asset(store: Path, content: str) -> str:
    encoded_content = content.encode()
    manifest = {"version": 1, "entries": [{
        "path": "index.txt", "type": "file", "size": len(encoded_content),
        "sha256": hashlib.sha256(encoded_content).hexdigest(), "executable": False,
    }]}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    directory = store / digest
    directory.mkdir(parents=True)
    (directory / "index.txt").write_bytes(encoded_content)
    (directory / ".arena-asset.json").write_bytes(encoded + b"\n")
    return digest


@pytest.fixture(autouse=True)
def receipt_ledger(tmp_path):
    root = tmp_path / "builds"
    root.mkdir(mode=0o700)
    for name in ("receipts", "bindings"):
        (root / name).mkdir(mode=0o700)
    path = root / "configuration.json"
    path.write_text(json.dumps({
        "version": 2, "input_assets_digest": "a" * 64, "environment_fingerprint": "b" * 64,
    }))
    path.chmod(0o600)
    return root


@pytest.fixture
def docker_commands(monkeypatch):
    commands = []

    def start(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO())

    def exchange(self, process, task):
        return json.dumps({"answer": task.decode()}).encode()

    monkeypatch.setattr("evaluator.runner.subprocess.Popen", start)
    monkeypatch.setattr(DockerRunner, "_exchange", exchange)
    return commands


def test_selected_assets_are_the_only_published_mounts(tmp_path, docker_commands):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    assets = tmp_path / "assets"
    (assets / "derived").mkdir(parents=True)
    store = tmp_path / "published"
    first = publish_asset(store, "first")
    second = publish_asset(store, "second")
    selected = sorted([first, second])
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        assets_dir=assets,
        published_assets_dir=store,
    )
    parameters = {"asset_digests": selected, "search": "ranked"}

    answer = json.loads(runner(candidate, "task", parameters=parameters).answer)
    first_command = docker_commands[-1]
    publish_asset(store, "later")
    runner(candidate, "task", parameters=parameters)
    mounts = [item for item in first_command if item.startswith("type=bind,")]

    assert mounts == [
        f"type=bind,src={candidate},dst=/candidate,readonly",
        f"type=bind,src={assets},dst=/assets,readonly",
        *(f"type=bind,src={store / digest},dst=/assets/derived/{digest},readonly"
          for digest in selected),
    ]
    assert [item for item in docker_commands[-1] if item.startswith("type=bind,")] == mounts
    assert "/assets/derived:rw,noexec,nosuid,nodev,size=1m,mode=0555,uid=0,gid=0" in first_command
    assert answer["parameters"] == parameters


@pytest.mark.parametrize("selection", [
    None, "a" * 64, {}, ("a" * 64,), [1], [True], ["A" * 64], ["g" * 64],
    ["a" * 63], ["../" + "a" * 64], ["a" * 64, "a" * 64], ["b" * 64, "a" * 64],
    [f"{index:064x}" for index in range(17)],
])
def test_invalid_asset_selection_fails_before_start(tmp_path, docker_commands, selection):
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task", parameters={"asset_digests": selection})

    assert docker_commands == []


@pytest.mark.parametrize("configured", [False, True])
def test_missing_store_fails_before_start(tmp_path, docker_commands, configured):
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=tmp_path / "missing" if configured else None,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task", parameters={"asset_digests": ["a" * 64]})

    assert docker_commands == []


def test_missing_selected_digest_fails_before_start(tmp_path, docker_commands):
    store = tmp_path / "store"
    store.mkdir()
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task", parameters={"asset_digests": ["a" * 64]})

    assert docker_commands == []


@pytest.mark.parametrize("mode", ["missing", "symlink", "file"])
def test_raw_assets_require_a_derived_mountpoint(tmp_path, docker_commands, mode):
    assets = tmp_path / "assets"
    assets.mkdir()
    if mode == "symlink":
        (assets / "derived").symlink_to(tmp_path)
    elif mode == "file":
        (assets / "derived").write_text("not a directory")
    store = tmp_path / "store"
    digest = publish_asset(store, "content")
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        assets_dir=assets,
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task", parameters={"asset_digests": [digest]})

    assert docker_commands == []


@pytest.mark.parametrize("selection", [{}, {"asset_digests": []}])
def test_without_selected_digests_published_store_is_not_mounted(
    tmp_path, docker_commands, selection,
):
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=tmp_path / "missing",
    )

    runner(tmp_path, "task", parameters=selection)

    assert docker_commands[0].count("--mount") == 1


@pytest.mark.parametrize("target", ["store", "directory", "manifest"])
def test_symlink_assets_are_rejected(tmp_path, docker_commands, target):
    store = tmp_path / "store"
    digest = publish_asset(store, "content")
    original = {"store": store, "directory": store / digest,
                "manifest": store / digest / ".arena-asset.json"}[target]
    moved = tmp_path / "moved"
    original.rename(moved)
    original.symlink_to(moved)
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task", parameters={"asset_digests": [digest]})

    assert docker_commands == []


@pytest.mark.parametrize("content", [b"{}\n", b"not json\n", b"[]\n"])
def test_invalid_asset_manifest_is_rejected(tmp_path, docker_commands, content):
    store = tmp_path / "store"
    digest = publish_asset(store, "content")
    (store / digest / ".arena-asset.json").write_bytes(content)
    runner = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task", parameters={"asset_digests": [digest]})

    assert docker_commands == []


def test_published_store_cannot_be_inside_unrestricted_assets(tmp_path):
    with pytest.raises(ValueError, match="asset"):
        DockerRunner(
            IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds",
            assets_dir=tmp_path,
            published_assets_dir=tmp_path / "derived",
        )


def test_published_store_is_part_of_frozen_runner_configuration(tmp_path):
    first = HarnessRunner(DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds",
        published_assets_dir=tmp_path / "first",
    ), {})
    second = HarnessRunner(DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds",
        published_assets_dir=tmp_path / "second",
    ), {})

    assert first.provenance()["default"]["published_assets_dir"] == str(tmp_path / "first")
    assert first.provenance()["digest"] != second.provenance()["digest"]


def test_harness_preflight_validates_assets_and_call_revalidates(tmp_path, docker_commands):
    store = tmp_path / "store"
    digest = publish_asset(store, "content")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "solution.json").write_text(json.dumps({
        "harness": "program", "parameters": {"asset_digests": [digest]},
    }))
    profile = DockerRunner(
        IMAGE, supports_asset_selection=True, build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )
    runner = HarnessRunner(profile, {"program": profile})

    runner.validate_candidate(candidate)
    assert docker_commands == []
    (store / digest / ".arena-asset.json").unlink()

    with pytest.raises(RunnerError, match="asset"):
        runner.validate_candidate(candidate)
    with pytest.raises(RunnerError, match="asset"):
        runner(candidate, "task")

    assert docker_commands == []


@pytest.mark.parametrize("selection", ["adapter-value", {"format": "local"}, ["not-a-digest"]])
def test_asset_selection_is_opaque_without_capability(tmp_path, docker_commands, selection):
    parameters = {"asset_digests": selection}
    runner = DockerRunner(IMAGE, uses_model=False, published_assets_dir=tmp_path / "missing")

    result = runner(tmp_path, "task", parameters=parameters)

    assert json.loads(result.answer)["parameters"] == parameters
    assert docker_commands[0].count("--mount") == 1
