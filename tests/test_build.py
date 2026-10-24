import base64
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evaluator.build import BuildStore, install_build_routes
from evaluator.build_receipts import bound_build_receipt
from evaluator.capacity import acquire_slot
from evaluator.contracts import Candidate
from evaluator.reliability import empty_usage
from evaluator.runner import RunnerError, SolveResult


def candidate(name="one", dependencies=()):
    files = {"build.py": base64.b64encode(name.encode()).decode()}
    if dependencies:
        files["solution.json"] = base64.b64encode(json.dumps({
            "harness": "program", "parameters": {"asset_digests": list(dependencies)},
        }).encode()).decode()
    return Candidate(id=name, files=files)


def store(tmp_path: Path, build=None, **kwargs):
    def default_build(candidate_dir, output):
        (output / "index.txt").write_text((candidate_dir / "build.py").read_text())
        return SolveResult("build complete", empty_usage(), 2.0)

    def dependency_assets(candidate_dir):
        manifest = candidate_dir / "solution.json"
        if not manifest.exists():
            return []
        return json.loads(manifest.read_text())["parameters"].get("asset_digests", [])

    kwargs.setdefault("dependency_assets", dependency_assets)
    return BuildStore(tmp_path / "builds", tmp_path / "assets", "a" * 64, "b" * 64,
                      build or default_build, **kwargs)


def build_once(builder, item):
    result, completion = builder.submit(item)
    if completion is not None:
        completion.result(timeout=10)
    return builder.read(result["candidate_digest"])


def test_build_runs_once_and_publishes_immutable_output(tmp_path):
    calls = []

    def build(candidate_dir, output):
        calls.append(candidate_dir)
        (output / "index.txt").write_text("prepared source")
        return SolveResult("build complete", empty_usage(), 2)

    builder = store(tmp_path, build)
    item = candidate()
    receipt = build_once(builder, item)
    assert receipt["status"] == "FINISHED"
    assert receipt["candidate_digest"] == item.digest()
    assert receipt["resources"]["solver"]["wall_seconds"] >= 2
    output = tmp_path / "assets" / receipt["output_digest"] / "index.txt"
    assert output.read_text() == "prepared source"
    assert output.stat().st_mode & 0o222 == 0
    assert build_once(builder, item) == receipt
    assert len(calls) == 1


def test_republishing_legacy_build_output_preserves_receipt_binding(tmp_path):
    builder = store(tmp_path)
    receipt = build_once(builder, candidate())
    legacy_id = "aa86f60b4495c5b3b6c5c30615b734b50b08a3b58e130d83296be92a3a52274e"
    original_id = receipt["output_digest"]
    (tmp_path / "assets" / original_id).rename(tmp_path / "assets" / legacy_id)
    (builder.root / "bindings" / f"{original_id}.json").rename(
        builder.root / "bindings" / f"{legacy_id}.json",
    )
    receipt["output_digest"] = legacy_id
    (builder.root / "receipts" / f"{receipt['candidate_digest']}.json").write_text(
        json.dumps(receipt),
    )
    source = builder.publisher.staging_root / "reimport"
    source.mkdir()
    (source / "index.txt").write_text("one")

    published = builder.publisher.publish("reimport")

    assert published["digest"] == legacy_id
    assert bound_build_receipt(builder.root, str(published["digest"])) == receipt
    dependent = build_once(builder, candidate("two", [published["digest"]]))
    assert dependent["dependency_receipts"] == [receipt["candidate_digest"]]
    assert dependent["resources"]["solver"]["wall_seconds"] == pytest.approx(
        receipt["own_resources"]["solver"]["wall_seconds"]
        + dependent["own_resources"]["solver"]["wall_seconds"],
    )


def test_failed_build_preserves_usage_without_retry(tmp_path):
    calls = []

    def fail(candidate_dir, output):
        calls.append(candidate_dir)
        raise RunnerError("failed", usage={**empty_usage(), "attempts": 2,
                                          "unreported_attempts": 2}, wall_seconds=7)

    builder = store(tmp_path, fail)
    receipt = build_once(builder, candidate())
    assert receipt["status"] == "FAILED"
    assert receipt["resources"]["solver"]["usage"]["unreported_attempts"] == 2
    assert receipt["resources"]["solver"]["wall_seconds"] >= 7
    assert build_once(builder, candidate())["status"] == "FAILED"
    assert len(calls) == 1


def test_dependency_costs_are_counted_once(tmp_path):
    builder = store(tmp_path)
    first = build_once(builder, candidate("first"))
    second = build_once(builder, candidate("second", [first["output_digest"]]))
    third = build_once(builder, candidate(
        "third", [first["output_digest"], second["output_digest"]],
    ))
    assert third["dependency_receipts"] == sorted([
        first["candidate_digest"], second["candidate_digest"],
    ])
    assert third["resources"]["solver"]["wall_seconds"] == pytest.approx(sum(
        row["own_resources"]["solver"]["wall_seconds"] for row in (first, second, third)
    ))


def test_capacity_rejection_does_not_create_receipt(tmp_path):
    slots = tmp_path / "slots"
    slots.mkdir(mode=0o700)
    builder = store(tmp_path, slot_dir=slots, slots=1)
    lease = acquire_slot(slots, 1)
    app = FastAPI()
    install_build_routes(app, builder)
    client = TestClient(app)
    try:
        response = client.post("/v1/builds", json={"candidate": candidate().model_dump()})
        assert response.status_code == 503
        assert not list((tmp_path / "builds" / "receipts").glob("*.json"))
    finally:
        assert lease is not None
        lease.close()
    response = client.post("/v1/builds", json={"candidate": candidate().model_dump()})
    assert response.status_code == 202
    receipt = client.get("/v1/builds/" + candidate().digest()).json()
    assert receipt["status"] == "FINISHED"


def test_environment_is_frozen(tmp_path):
    store(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        BuildStore(tmp_path / "builds", tmp_path / "assets", "c" * 64, "b" * 64,
                   lambda candidate, output: SolveResult("", empty_usage(), 0))


def test_build_does_not_interpret_adapter_parameters_without_asset_capability(tmp_path):
    item = candidate("custom")
    item.files["solution.json"] = base64.b64encode(json.dumps({
        "harness": "custom", "parameters": {"asset_digests": "private-adapter-setting"},
    }).encode()).decode()
    builder = store(tmp_path, dependency_assets=None)

    receipt = build_once(builder, item)

    assert receipt["status"] == "FINISHED"
    assert receipt["dependency_receipts"] == []
    assert receipt["version"] == 2
    assert receipt["input_assets_digest"] == "a" * 64
    assert "input_kb_digest" not in receipt


def test_rejected_build_harness_returns_422_without_reserving_candidate(tmp_path):
    checks = []

    def unsupported(candidate_dir):
        checks.append(candidate_dir)
        raise RunnerError("selected harness does not support trusted builds")

    builder = store(tmp_path, dependency_assets=unsupported)
    app = FastAPI()
    install_build_routes(app, builder)
    client = TestClient(app)

    for _ in range(2):
        response = client.post("/v1/builds", json={"candidate": candidate().model_dump()})
        assert response.status_code == 422
    assert len(checks) == 2
    assert list((builder.root / "jobs").iterdir()) == []
    assert list((builder.root / "receipts").iterdir()) == []
