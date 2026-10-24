import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluator.contracts import Example, Metric
from evaluator.metric_package import freeze_metric, load_metric, metric_package_provenance
from evaluator.metrics import command_provenance, evaluate_answer


def draft_metric(tmp_path: Path) -> Path:
    directory = tmp_path / "draft"
    directory.mkdir()
    (directory / "metric.json").write_text(json.dumps({
        "version": "synthetic-v1", "kind": "command",
        "command": {"argv": ["python", "score.py", "policy.json"]},
    }))
    (directory / "score.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "record = json.load(sys.stdin)\n"
        "policy = json.loads(Path(sys.argv[1]).read_text())\n"
        "print(json.dumps({'score': len(record['answer']) * policy['weight']}))\n"
    )
    (directory / "policy.json").write_text('{"weight": 2}')
    (directory / "description.md").write_text("Synthetic length metric.\n")
    return directory


def test_frozen_package_runs_program_metric_and_lists_all_files(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_metric(source, frozen)
    metric = load_metric(frozen)
    assert metric.command is not None
    assert metric.command.argv == [
        str(Path(sys.executable).absolute()), str(frozen / "score.py"),
        str(frozen / "policy.json"),
    ]
    provenance = metric_package_provenance(frozen)
    assert provenance == {
        "format_version": 1,
        "files": {
            name: {"executable": False}
            for name in ("metric.json", "score.py", "policy.json", "description.md")
        },
    }
    assert command_provenance(metric)
    assert evaluate_answer(metric, Example(id="x", input="synthetic"), "abc")["score"] == 6


def test_package_inventory_is_independent_of_installation_path(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    freeze_metric(draft_metric(tmp_path), frozen)
    relocated = tmp_path / "relocated"
    shutil.copytree(frozen, relocated)
    assert metric_package_provenance(frozen) == metric_package_provenance(relocated)
    metric = load_metric(relocated)
    assert metric.command is not None
    assert metric.command.argv[1] == str(relocated / "score.py")


def test_package_content_is_not_checked_against_a_checksum(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    freeze_metric(draft_metric(tmp_path), frozen)
    original = metric_package_provenance(frozen)
    (frozen / "policy.json").write_text('{"weight": 100}')
    metric = load_metric(frozen)
    assert metric_package_provenance(frozen) == original
    assert evaluate_answer(metric, Example(id="x", input="synthetic"), "abc")["score"] == 300


@pytest.mark.parametrize("change", ["add", "delete", "executable"])
def test_changed_frozen_package_inventory_is_rejected(tmp_path: Path, change: str) -> None:
    frozen = tmp_path / "frozen"
    freeze_metric(draft_metric(tmp_path), frozen)
    asset = frozen / "policy.json"
    if change == "add":
        (frozen / "new.py").write_text("pass\n")
    elif change == "delete":
        asset.unlink()
    else:
        asset.chmod(0o755)
    with pytest.raises(ValueError, match="lock"):
        load_metric(frozen)
    with pytest.raises(ValueError, match="lock"):
        metric_package_provenance(frozen)


def test_standalone_metric_loads_without_package(tmp_path: Path) -> None:
    path = tmp_path / "metric.json"
    executable = str(Path(sys.executable).absolute())
    metric = Metric.model_validate({
        "version": "legacy", "kind": "command",
        "command": {
            "argv": [executable, "-c", "print('{\"score\": 1}')"],
        },
    })
    path.write_text(metric.model_dump_json())
    assert load_metric(path) == metric
    assert metric_package_provenance(path) is None


def test_legacy_standalone_metric_ignores_obsolete_source_checksums(tmp_path: Path) -> None:
    path = tmp_path / "metric.json"
    path.write_text(json.dumps({
        "version": "legacy", "kind": "command",
        "command": {
            "argv": [sys.executable, "-c", "print('{\"score\": 1}')"],
            "source_sha256": {sys.executable: "invalid"},
        },
    }))
    metric = load_metric(path)
    assert metric.command is not None
    assert "source_sha256" not in metric.command.model_dump()
    assert evaluate_answer(metric, Example(id="x", input=""), "")["score"] == 1


@pytest.mark.parametrize("legacy_checksum", [None, "invalid", {"invalid": []}])
def test_legacy_package_lock_ignores_obsolete_checksums(
    tmp_path: Path, legacy_checksum: object,
) -> None:
    frozen = tmp_path / "frozen"
    original = freeze_metric(draft_metric(tmp_path), frozen)
    lock = frozen / "metric.lock.json"
    content = json.loads(lock.read_text())
    for entry in content["files"].values():
        entry["sha256"] = legacy_checksum
    lock.write_text(json.dumps(content))
    (frozen / "policy.json").write_text('{"weight": 100}')
    metric = load_metric(frozen)
    assert metric_package_provenance(frozen) == original
    assert evaluate_answer(metric, Example(id="x", input=""), "abc")["score"] == 300


def test_unsealed_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lock"):
        load_metric(draft_metric(tmp_path))


@pytest.mark.parametrize("name", [".git", ".env", ".env.local"])
def test_restricted_files_are_rejected(tmp_path: Path, name: str) -> None:
    source = draft_metric(tmp_path)
    (source / name).write_text("not a real secret")
    with pytest.raises(ValueError, match="restricted"):
        freeze_metric(source, tmp_path / "frozen")


def test_symlink_and_special_file_are_rejected(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    link = source / "link"
    link.symlink_to(source / "score.py")
    with pytest.raises(ValueError, match="symlink"):
        freeze_metric(source, tmp_path / "frozen")
    link.unlink()
    os.mkfifo(source / "fifo")
    with pytest.raises(ValueError, match="regular"):
        freeze_metric(source, tmp_path / "frozen")


def test_symlink_package_root_is_rejected(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        freeze_metric(link, tmp_path / "frozen")


def test_unreadable_subdirectory_fails_without_incomplete_output(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    hidden = source / "nested"
    hidden.mkdir()
    (hidden / "data.json").write_text("{}")
    scandir = os.scandir

    def guarded_scan(
        path: str | os.PathLike[str],
    ) -> AbstractContextManager[Iterator[os.DirEntry[str]]]:
        if Path(path) == hidden:
            raise PermissionError("unreadable")
        return scandir(path)

    with patch("evaluator.filesystem.os.scandir", guarded_scan):
        with pytest.raises(PermissionError, match="unreadable"):
            freeze_metric(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_freeze_does_not_overwrite_or_copy_into_source(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        freeze_metric(source, source / "nested")
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        freeze_metric(source, existing)
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize("argument", ["../outside.py", "/tmp/outside.py"])
def test_package_command_paths_must_be_portable(tmp_path: Path, argument: str) -> None:
    source = draft_metric(tmp_path)
    config = json.loads((source / "metric.json").read_text())
    config["command"]["argv"][1] = argument
    (source / "metric.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="relative"):
        freeze_metric(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_freeze_validates_without_executing_metric(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    (source / "score.py").write_text("raise RuntimeError('must not execute')\n")
    freeze_metric(source, tmp_path / "frozen")
    assert load_metric(tmp_path / "frozen").kind == "command"


def test_invalid_config_does_not_leave_output(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    (source / "metric.json").write_text('{"version": "invalid"}')
    with pytest.raises(ValueError):
        freeze_metric(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_aggregation_and_packaged_executable_are_resolved(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    script = source / "aggregate"
    script.write_text("#!/bin/sh\nprintf '{\"score\": 1}'\n")
    script.chmod(0o755)
    config = json.loads((source / "metric.json").read_text())
    config["aggregation"] = {"argv": ["./aggregate"]}
    (source / "metric.json").write_text(json.dumps(config))
    frozen = tmp_path / "frozen"
    freeze_metric(source, frozen)
    metric = load_metric(frozen)
    assert metric.aggregation is not None
    assert metric.aggregation.argv == [str(frozen / "aggregate")]


def test_frozen_lock_has_strict_schema(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    freeze_metric(draft_metric(tmp_path), frozen)
    lock = frozen / "metric.lock.json"
    content = json.loads(lock.read_text())
    content["extra"] = "invalid"
    lock.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_metric(frozen)


def test_interpreter_invocation_preserves_virtual_environment_path(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    frozen = tmp_path / "frozen"
    with patch("evaluator.metric_package.sys.executable", str(executable)):
        freeze_metric(source, frozen)
        metric = load_metric(frozen)
    assert metric.command is not None
    assert metric.command.argv[0] == str(executable)


def test_builtin_llm_configuration_can_be_frozen_without_model_calls(tmp_path: Path) -> None:
    source = tmp_path / "draft"
    source.mkdir()
    (source / "metric.json").write_text(json.dumps({
        "version": "synthetic-reference-v1", "instruction": "Judge synthetic examples.",
        "criteria": [{"id": "correct", "question": "Correct?", "weight": 1}],
    }))
    frozen = tmp_path / "frozen"
    freeze_metric(source, frozen)
    assert load_metric(frozen).kind == "llm"


def test_copy_error_does_not_publish_partial_package(tmp_path: Path) -> None:
    source = draft_metric(tmp_path)
    output = tmp_path / "frozen"
    with patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            freeze_metric(source, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]
