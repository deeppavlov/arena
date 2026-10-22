import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.build_receipts import (
    bound_build_receipt,
    build_environment,
    load_build_receipt,
    read_receipt,
    validate_build_environment,
)
from evaluator.client import pack_candidate
from evaluator.reliability import empty_usage


@pytest.fixture
def build_store(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "build.py").write_text("print('synthetic')\n")
    digest = pack_candidate("build", candidate).digest()
    root = tmp_path / "builds"
    root.mkdir(mode=0o700)
    (root / "receipts").mkdir(mode=0o700)
    configuration = {
        "version": 2, "input_assets_digest": "a" * 64, "environment_fingerprint": "b" * 64,
    }
    config_path = root / "configuration.json"
    config_path.write_text(json.dumps(configuration))
    config_path.chmod(0o600)
    resources = {
        "solver": {"usage": empty_usage(), "wall_seconds": 1.5, "usage_complete": True},
    }
    receipt = {
        **configuration, "status": "FINISHED", "candidate_digest": digest,
        "output_digest": "c" * 64, "resources": resources,
        "own_resources": deepcopy(resources), "dependency_receipts": ["d" * 64],
        "trace": {"details": "optional"},
    }
    path = root / "receipts" / f"{digest}.json"
    path.write_text(json.dumps(receipt))
    path.chmod(0o600)
    return root, candidate, digest, receipt, path


def write_receipt(store, receipt):
    store[-1].write_text(json.dumps(receipt))


def test_receipt_is_optional_without_build(tmp_path):
    assert load_build_receipt(None, tmp_path) is None


def test_load_receipt_matches_candidate_and_frozen_configuration(build_store):
    root, candidate, digest, receipt, _ = build_store
    assert load_build_receipt(root, candidate) == receipt
    assert load_build_receipt(root, candidate, digest) == receipt
    assert read_receipt(root, digest) == receipt


def test_build_requires_store_and_matching_receipt(build_store):
    root, candidate, _, _, path = build_store
    with pytest.raises(ValueError, match="build receipt"):
        load_build_receipt(None, candidate)
    path.unlink()
    with pytest.raises(ValueError, match="build receipt"):
        load_build_receipt(root, candidate)


@pytest.mark.parametrize(("field", "value"), [
    ("version", True), ("version", 1), ("status", "UNKNOWN"), ("status", "FAILED"),
    ("candidate_digest", "e" * 64), ("input_assets_digest", "e" * 64),
    ("environment_fingerprint", "e" * 64), ("output_digest", "invalid"),
    ("dependency_receipts", ["d" * 64, "d" * 64]),
    ("dependency_receipts", ["../outside"]), ("dependency_receipts", "d" * 64),
    ("resources", None), ("own_resources", {}),
])
def test_rejects_malformed_or_stale_receipt(build_store, field, value):
    root, _, digest, receipt, _ = build_store
    receipt[field] = value
    write_receipt(build_store, receipt)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


@pytest.mark.parametrize(("field", "value"), [
    ("wall_seconds", -1), ("wall_seconds", float("nan")),
    ("wall_seconds", float("inf")), ("wall_seconds", True), ("wall_seconds", "1"),
    ("usage_complete", 1), ("usage_complete", None),
])
@pytest.mark.parametrize("block", ["resources", "own_resources"])
def test_rejects_invalid_resource_values(build_store, field, value, block):
    root, _, digest, receipt, _ = build_store
    receipt[block]["solver"][field] = value
    write_receipt(build_store, receipt)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


@pytest.mark.parametrize("usage", [
    {}, {**empty_usage(), "attempts": True}, {**empty_usage(), "attempts": -1},
    {**empty_usage(), "unreported_attempts": 1},
    {**empty_usage(), "prompt_tokens": 1},
    {**empty_usage(), "prompt_tokens": 1, "total_tokens": 1},
])
def test_rejects_invalid_usage(build_store, usage):
    root, _, digest, receipt, _ = build_store
    receipt["resources"]["solver"]["usage"] = usage
    write_receipt(build_store, receipt)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


def test_accepts_incomplete_accounting_without_claiming_completeness(build_store):
    root, _, digest, receipt, _ = build_store
    receipt["resources"]["solver"].update(
        usage={**empty_usage(), "attempts": 2, "unreported_attempts": 1},
        usage_complete=False,
    )
    write_receipt(build_store, receipt)
    assert read_receipt(root, digest)["resources"]["solver"]["usage_complete"] is False


@pytest.mark.parametrize("state", ["RUNNING", "FAILED"])
def test_build_store_can_read_incomplete_receipts(build_store, state):
    root, _, digest, receipt, _ = build_store
    receipt["status"] = state
    for key in ("output_digest", "resources", "own_resources"):
        receipt.pop(key)
    write_receipt(build_store, receipt)
    assert read_receipt(root, digest, require_complete=False) == receipt
    with pytest.raises(ValueError):
        read_receipt(root, digest)


@pytest.mark.parametrize("relative", [".", "receipts", "configuration.json", "receipt"])
def test_rejects_writable_storage(build_store, relative):
    root, _, digest, _, path = build_store
    target = path if relative == "receipt" else root / relative
    target.chmod(0o777 if target.is_dir() else 0o666)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


@pytest.mark.parametrize("relative", [".", "receipts", "configuration.json", "receipt"])
def test_rejects_symlink_storage(build_store, relative):
    root, _, digest, _, path = build_store
    target = path if relative == "receipt" else root / relative
    original = target.with_name(target.name + "-original")
    target.rename(original)
    target.symlink_to(original)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


def test_rejects_nonregular_receipt(build_store):
    root, _, digest, _, path = build_store
    path.unlink()
    os.mkfifo(path, 0o600)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


def test_rejects_untrusted_owner(build_store, monkeypatch):
    root, _, digest, _, _ = build_store
    monkeypatch.setattr("evaluator.build_receipts.os.geteuid", lambda: os.getuid() + 1)
    with pytest.raises(ValueError):
        read_receipt(root, digest)


@pytest.mark.parametrize("digest", ["../outside", "", "A" * 64, None])
def test_rejects_invalid_receipt_digest(build_store, digest):
    with pytest.raises(ValueError):
        read_receipt(build_store[0], digest)


def test_rejects_missing_or_malformed_configuration(build_store):
    root, _, digest, _, _ = build_store
    configuration = root / "configuration.json"
    configuration.write_text("{}")
    with pytest.raises(ValueError):
        read_receipt(root, digest)
    configuration.unlink()
    with pytest.raises(ValueError):
        read_receipt(root, digest)


def bind_output(store):
    root, _, digest, receipt, _ = store
    bindings = root / "bindings"
    bindings.mkdir(mode=0o700)
    path = bindings / f"{receipt['output_digest']}.json"
    path.write_text(json.dumps({"candidate_digest": digest}))
    path.chmod(0o600)
    return path


def test_bound_receipt_is_optional_before_build(build_store, tmp_path):
    root, _, _, receipt, _ = build_store
    output = receipt["output_digest"]
    for unavailable in (None, tmp_path / "missing", root):
        with pytest.raises(ValueError):
            bound_build_receipt(unavailable, output)
    (root / "bindings").mkdir(mode=0o700)
    assert bound_build_receipt(root, output) is None


def test_bound_receipt_resolves_finished_build(build_store):
    root, _, _, receipt, _ = build_store
    bind_output(build_store)
    assert bound_build_receipt(root, receipt["output_digest"]) == receipt


@pytest.mark.parametrize("mapping", [None, {}, {"candidate_digest": "invalid"},
                                     {"candidate_digest": "f" * 64}])
def test_bound_receipt_rejects_malformed_or_missing_build(build_store, mapping):
    root, _, _, receipt, _ = build_store
    path = bind_output(build_store)
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):
        bound_build_receipt(root, receipt["output_digest"])


@pytest.mark.parametrize("target", ["root", "bindings", "file"])
def test_bound_receipt_rejects_symlinks(build_store, target):
    root, _, _, receipt, _ = build_store
    path = bind_output(build_store)
    path = {"root": root, "bindings": path.parent, "file": path}[target]
    original = path.with_name(path.name + "-original")
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError):
        bound_build_receipt(root, receipt["output_digest"])


def test_bound_receipt_rejects_writable_binding(build_store):
    root, _, _, receipt, _ = build_store
    bind_output(build_store).chmod(0o666)
    with pytest.raises(ValueError):
        bound_build_receipt(root, receipt["output_digest"])


def test_bound_receipt_rejects_unfinished_build(build_store):
    root, _, _, receipt, _ = build_store
    bind_output(build_store)
    receipt["status"] = "FAILED"
    write_receipt(build_store, receipt)
    with pytest.raises(ValueError):
        bound_build_receipt(root, receipt["output_digest"])


def test_bound_receipt_rejects_different_output(build_store):
    root, _, _, receipt, _ = build_store
    bind_output(build_store)
    output = receipt["output_digest"]
    receipt["output_digest"] = "e" * 64
    write_receipt(build_store, receipt)
    with pytest.raises(ValueError):
        bound_build_receipt(root, output)


@pytest.fixture
def harness_runner():
    default = SimpleNamespace(
        image="sha256:" + "a" * 64, model="internal", uses_model=True,
        timeout=100, model_timeout=20, retry_delay=1, prepare_timeout=30,
        supports_build=False, supports_asset_selection=False,
        memory="2g", cpus="2", pids_limit=64, tmpfs_size="128m", tmpfs_exec=False,
        assets_dir=Path("/train/assets"), usage_dir=Path("/train/usage"),
        published_assets_dir=Path("/train/published"), build_dir=Path("/train/builds"),
        build_output_dir=Path("/train/output"), network="train-network", split="train",
    )
    return SimpleNamespace(default_runner=default, registry={"alternative": deepcopy(default)})


@pytest.mark.parametrize("field", [
    "assets_dir", "usage_dir", "published_assets_dir", "build_dir", "build_output_dir",
    "network", "split",
])
def test_build_environment_ignores_deployment_paths(harness_runner, field):
    before = build_environment(harness_runner, {"temperature": 0})
    setattr(harness_runner.default_runner, field, "/private/different")
    setattr(harness_runner.registry["alternative"], field, "/private/other")
    assert build_environment(harness_runner, {"temperature": 0}) == before


@pytest.mark.parametrize(("field", "value"), [
    ("image", "sha256:" + "b" * 64), ("model", "other-model"), ("uses_model", False),
    ("timeout", 101), ("model_timeout", 21), ("retry_delay", 2), ("prepare_timeout", 31),
    ("memory", "4g"), ("cpus", "4"), ("pids_limit", 65), ("tmpfs_size", "256m"),
    ("tmpfs_exec", True), ("supports_build", True), ("supports_asset_selection", True),
])
@pytest.mark.parametrize("selected", ["default", "alternative"])
def test_build_environment_pins_execution_settings(harness_runner, field, value, selected):
    before = build_environment(harness_runner, {})
    runner = harness_runner.default_runner if selected == "default" else (
        harness_runner.registry[selected]
    )
    setattr(runner, field, value)
    assert build_environment(harness_runner, {}) != before


def test_build_environment_pins_parameters_and_harness_names(harness_runner):
    before = build_environment(harness_runner, {"a": 1, "b": 2})
    assert build_environment(harness_runner, {"b": 2, "a": 1}) == before
    assert build_environment(harness_runner, {"a": 1, "b": 3}) != before
    harness_runner.registry["renamed"] = harness_runner.registry.pop("alternative")
    assert build_environment(harness_runner, {"a": 1, "b": 2}) != before


def test_build_environment_keeps_default_separate_from_registry(harness_runner):
    harness_runner.registry["default"] = harness_runner.registry["alternative"]
    before = build_environment(harness_runner, {})
    harness_runner.default_runner.model = "changed-default"
    assert build_environment(harness_runner, {}) != before


def test_validate_build_environment_matches_frozen_configuration(build_store):
    root, _, _, receipt, _ = build_store
    validate_build_environment(None, "unconfigured")
    validate_build_environment(root, receipt["environment_fingerprint"])
    validate_build_environment(
        root, receipt["environment_fingerprint"], receipt["input_assets_digest"],
    )


@pytest.mark.parametrize(("environment", "input_assets"), [
    ("e" * 64, None), ("invalid", None), ("b" * 64, "e" * 64), ("b" * 64, "invalid"),
])
def test_validate_build_environment_rejects_mismatch(build_store, environment, input_assets):
    with pytest.raises(ValueError):
        validate_build_environment(build_store[0], environment, input_assets)


@pytest.mark.parametrize("problem", ["missing", "symlink", "writable", "malformed"])
def test_validate_build_environment_requires_trusted_configuration(build_store, problem):
    root, _, _, receipt, _ = build_store
    path = root / "configuration.json"
    if problem == "missing":
        path.unlink()
    elif problem == "symlink":
        original = path.with_name("original.json")
        path.rename(original)
        path.symlink_to(original)
    elif problem == "writable":
        path.chmod(0o666)
    else:
        path.write_text("{}")
    with pytest.raises(ValueError):
        validate_build_environment(root, receipt["environment_fingerprint"])
