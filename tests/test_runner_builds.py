import json

import pytest

from evaluator.runner import DockerRunner, HarnessRunner, RunnerError
from tests.test_runner_assets import IMAGE, publish_asset
from tests.test_runner_assets import docker_commands as docker_commands
from tests.test_runner_assets import receipt_ledger as receipt_ledger


@pytest.fixture
def preparation(monkeypatch):
    calls = []
    response = {"receipt": None}

    def load(root, candidate_dir, candidate_digest=None):
        calls.append((root, candidate_dir))
        if isinstance(response["receipt"], Exception):
            raise response["receipt"]
        return response["receipt"]

    monkeypatch.setattr("evaluator.runner.load_build_receipt", load)
    return calls, response


def test_trusted_build_mounts_output_and_injects_mode(
    tmp_path, docker_commands, preparation,
):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "build.py").write_text("print('build')")
    output = tmp_path / "output"
    output.mkdir()
    calls, response = preparation
    response["receipt"] = ValueError("receipt does not exist")
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        build_output_dir=output,
    )
    parameters = {"search": "ranked"}

    result = json.loads(runner(candidate, "", parameters=parameters).answer)

    assert result["operation"] == "build"
    assert result["parameters"] == {"search": "ranked"}
    assert "input" not in result
    assert parameters == {"search": "ranked"}
    assert calls == []
    assert f"type=bind,src={output},dst=/output" in docker_commands[0]
    assert not any("dst=/assets/prepared" in item for item in docker_commands[0])


def test_trusted_build_does_not_create_parameters_when_missing(
    tmp_path, docker_commands, preparation,
):
    result = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        build_output_dir=tmp_path,
    )(tmp_path, "")

    assert json.loads(result.answer)["operation"] == "build"
    assert "parameters" not in json.loads(result.answer)


def test_trusted_build_rejects_example_input(tmp_path, docker_commands, preparation):
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        build_output_dir=tmp_path,
    )

    with pytest.raises(RunnerError, match="build input"):
        runner(tmp_path, "private question")

    assert docker_commands == []


@pytest.mark.parametrize("value", [True, False, None, 0, "true"])
def test_adapter_parameter_cannot_enable_trusted_build(
    tmp_path, docker_commands, preparation, value,
):
    runner = DockerRunner(IMAGE, uses_model=False)
    result = runner(tmp_path, "task", parameters={"_arena_build": value})

    assert json.loads(result.answer)["parameters"] == {"_arena_build": value}
    assert json.loads(result.answer).get("operation", "solve") == "solve"
    assert not any("dst=/output" in item for item in docker_commands[0])


def test_arbitrary_parameters_cannot_mount_output(tmp_path, docker_commands, preparation):
    result = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
    )(
        tmp_path, "task", parameters={"build_output_dir": str(tmp_path)},
    )

    assert "_arena_build" not in json.loads(result.answer)["parameters"]
    assert not any("dst=/output" in item for item in docker_commands[0])


def test_known_prepared_import_requires_a_metered_build(
    tmp_path, docker_commands, preparation, monkeypatch,
):
    store = tmp_path / "published"
    digest = publish_asset(store, "existing preparation")
    monkeypatch.setattr("evaluator.runner.bound_build_receipt", lambda root, value: {
        "candidate_digest": "a" * 64, "output_digest": value,
    })
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="requires build.py"):
        runner(tmp_path, "task", parameters={"asset_digests": [digest]})

    assert docker_commands == []


def test_unknown_import_remains_available_without_build(
    tmp_path, docker_commands, preparation, monkeypatch,
):
    store = tmp_path / "published"
    digest = publish_asset(store, "downloaded weights")
    monkeypatch.setattr("evaluator.runner.bound_build_receipt", lambda root, value: None)
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )

    runner(tmp_path, "task", parameters={"asset_digests": [digest]})

    assert f"type=bind,src={store / digest},dst=/assets/derived/{digest},readonly" in (
        docker_commands[0]
    )


def test_prepared_output_comes_from_receipt_and_keeps_costs_out_of_per_example_usage(
    tmp_path, docker_commands, preparation,
):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    assets = tmp_path / "assets"
    (assets / "prepared").mkdir(parents=True)
    store = tmp_path / "published"
    digest = publish_asset(store, "prepared representation")
    build_dir = tmp_path / "builds"
    calls, response = preparation
    response["receipt"] = {
        "output_digest": digest,
        "resources": {"solver": {"usage": {"total_tokens": 250}, "wall_seconds": 75}},
    }
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        uses_model=False, assets_dir=assets, published_assets_dir=store,
        build_dir=build_dir,
    )

    result = runner(candidate, "task")

    assert calls == [(build_dir, candidate)]
    assert f"type=bind,src={store / digest},dst=/assets/prepared,readonly" in docker_commands[0]
    assert not any("dst=/output" in item for item in docker_commands[0])
    assert result.usage["total_tokens"] == 0
    assert result.wall_seconds < 75


def test_missing_receipt_rejected_before_admission_or_execution(
    tmp_path, docker_commands, preparation,
):
    _, response = preparation
    response["receipt"] = ValueError("build is incomplete")
    profile = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True, uses_model=False,
        build_dir=tmp_path / "builds",
    )
    runner = HarnessRunner(profile, {})

    with pytest.raises(RunnerError, match="preparation"):
        runner.validate_candidate(tmp_path)
    with pytest.raises(RunnerError, match="preparation"):
        runner(tmp_path, "task")

    assert docker_commands == []


def test_receipt_is_revalidated_between_admission_and_execution(
    tmp_path, docker_commands, preparation,
):
    store = tmp_path / "published"
    digest = publish_asset(store, "prepared")
    _, response = preparation
    response["receipt"] = {"output_digest": digest}
    profile = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )
    runner = HarnessRunner(profile, {})
    runner.validate_candidate(tmp_path)
    response["receipt"] = ValueError("environment changed")

    with pytest.raises(RunnerError, match="preparation"):
        runner(tmp_path, "task")

    assert docker_commands == []


@pytest.mark.parametrize("failure", ["missing", "manifest", "symlink"])
def test_prepared_output_must_be_valid_published_asset(
    tmp_path, docker_commands, preparation, failure,
):
    store = tmp_path / "published"
    digest = publish_asset(store, "prepared")
    _, response = preparation
    response["receipt"] = {"output_digest": digest}
    manifest = store / digest / ".arena-asset.json"
    if failure == "missing":
        manifest.unlink()
    elif failure == "manifest":
        manifest.write_text("{}\n")
    else:
        moved = tmp_path / "moved"
        (store / digest).rename(moved)
        (store / digest).symlink_to(moved)
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", uses_model=False,
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="asset"):
        runner(tmp_path, "task")

    assert docker_commands == []


@pytest.mark.parametrize("mode", ["missing", "symlink", "file"])
def test_prepared_mountpoint_is_validated(tmp_path, docker_commands, preparation, mode):
    assets = tmp_path / "assets"
    assets.mkdir()
    if mode == "symlink":
        (assets / "prepared").symlink_to(tmp_path)
    elif mode == "file":
        (assets / "prepared").write_text("not a directory")
    store = tmp_path / "published"
    digest = publish_asset(store, "prepared")
    _, response = preparation
    response["receipt"] = {"output_digest": digest}
    runner = DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True, build_dir=tmp_path / "builds",
        uses_model=False, assets_dir=assets, published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="mountpoint"):
        runner(tmp_path, "task")

    assert docker_commands == []


def test_build_configuration_changes_frozen_runner_identity(tmp_path, preparation):
    original = HarnessRunner(DockerRunner(IMAGE), {}).provenance()
    configured = HarnessRunner(DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True, build_dir=tmp_path,
    ), {}).provenance()
    trusted = HarnessRunner(DockerRunner(
        IMAGE, supports_build=True, supports_asset_selection=True,
        build_dir=tmp_path / "builds", build_output_dir=tmp_path,
    ), {}).provenance()

    assert original["digest"] != configured["digest"] != trusted["digest"]
    assert configured["default"]["build_dir"] == str(tmp_path)
    assert trusted["default"]["build_output_dir"] == str(tmp_path)


def test_unrelated_build_file_is_opaque_without_capability(tmp_path, docker_commands):
    (tmp_path / "build.py").write_text("raise AssertionError('adapter-owned helper')")
    runner = HarnessRunner(DockerRunner(IMAGE, uses_model=False), {})

    runner.validate_candidate(tmp_path)
    result = runner(tmp_path, "task")

    assert json.loads(result.answer)["input"] == "task"
    assert runner.preparation_receipt(tmp_path) is None
    assert docker_commands[0].count("--mount") == 1


def test_build_rejects_unsupported_selected_harness_before_start(tmp_path, docker_commands):
    from evaluator.build import run_build

    (tmp_path / "solution.json").write_text(json.dumps({"harness": "legacy"}))
    runner = HarnessRunner(
        DockerRunner(IMAGE, uses_model=False, supports_build=True),
        {"legacy": DockerRunner(IMAGE, uses_model=False)},
    )

    with pytest.raises(RunnerError, match="does not support"):
        run_build(runner, tmp_path, tmp_path)
    with pytest.raises(RunnerError, match="does not support"):
        runner.build_asset_digests(tmp_path)
    assert docker_commands == []


def test_build_selected_harness_uses_top_level_operation(tmp_path, docker_commands):
    from evaluator.build import run_build

    (tmp_path / "solution.json").write_text(json.dumps({
        "harness": "program", "parameters": {"_arena_build": "adapter-value"},
    }))
    runner = HarnessRunner(
        DockerRunner(IMAGE, uses_model=False),
        {"program": DockerRunner(IMAGE, uses_model=False, supports_build=True)},
    )

    result = run_build(runner, tmp_path, tmp_path)

    task = json.loads(result.answer)
    assert task["operation"] == "build"
    assert task["parameters"] == {"_arena_build": "adapter-value"}
    assert "input" not in task
    assert runner.build_asset_digests(tmp_path) == []


@pytest.mark.parametrize("field", ["supports_build", "supports_asset_selection"])
@pytest.mark.parametrize("value", [1, "true", None])
def test_capability_flags_must_be_booleans(field, value):
    with pytest.raises(ValueError, match="capabilit"):
        DockerRunner(IMAGE, **{field: value})


def test_asset_only_harness_cannot_import_unaccounted_preparation(
    tmp_path, docker_commands, monkeypatch,
):
    store = tmp_path / "published"
    digest = publish_asset(store, "existing preparation")
    monkeypatch.setattr("evaluator.runner.bound_build_receipt", lambda root, value: {
        "candidate_digest": "a" * 64, "output_digest": value,
    })
    runner = DockerRunner(
        IMAGE, uses_model=False, supports_asset_selection=True, build_dir=tmp_path / "builds",
        published_assets_dir=store,
    )

    with pytest.raises(RunnerError, match="trusted build support"):
        runner(tmp_path, "task", parameters={"asset_digests": [digest]})

    assert docker_commands == []


@pytest.mark.parametrize("ledger", ["unset", "missing", "missing-config", "missing-bindings",
                                    "malformed", "configured"])
def test_real_metered_output_never_becomes_free_when_receipt_ledger_is_unavailable(
    tmp_path, docker_commands, ledger,
):
    from evaluator.build import BuildStore
    from evaluator.client import pack_candidate
    from evaluator.reliability import empty_usage
    from evaluator.runner import SolveResult

    source = tmp_path / "source"
    source.mkdir()
    (source / "build.py").write_text("prepared source")
    published = tmp_path / "published"

    def build(candidate_dir, output):
        (output / "index.txt").write_text("measured representation")
        return SolveResult("build complete", empty_usage(), 7)

    builder = BuildStore(tmp_path / "ledger", published, "a" * 64, "b" * 64, build)
    receipt, completion = builder.submit(pack_candidate("prepared", source))
    assert completion is not None
    completion.result(timeout=10)
    receipt = builder.read(receipt["candidate_digest"])
    assert receipt["status"] == "FINISHED"
    assert receipt["resources"]["solver"]["wall_seconds"] >= 7
    candidate = tmp_path / "consumer"
    candidate.mkdir()
    (candidate / "solution.json").write_text(json.dumps({
        "harness": "assets", "parameters": {"asset_digests": [receipt["output_digest"]]},
    }))
    if ledger == "unset":
        with pytest.raises(ValueError, match="receipt ledger"):
            DockerRunner(IMAGE, uses_model=False, supports_asset_selection=True,
                         published_assets_dir=published)
        return
    root = builder.root
    if ledger == "missing":
        root = tmp_path / "unavailable"
    elif ledger == "missing-config":
        (root / "configuration.json").unlink()
    elif ledger == "missing-bindings":
        (root / "bindings").rename(root / "unavailable-bindings")
    elif ledger == "malformed":
        (root / "configuration.json").write_text("{}")
    profile = DockerRunner(IMAGE, uses_model=False, supports_asset_selection=True,
                           published_assets_dir=published, build_dir=root)
    runner = HarnessRunner(profile, {"assets": profile})

    with pytest.raises(RunnerError):
        runner.validate_candidate(candidate)
    with pytest.raises(RunnerError):
        runner(candidate, "task")
    assert docker_commands == []
