import os
import stat
import subprocess
from pathlib import Path

import pytest

from evaluator.candidates import materialize_candidate, materialize_files
from evaluator.client import pack_candidate
from evaluator.contracts import Candidate


def test_candidate_roundtrip_runs_nested_helper_with_canonical_modes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested" / "bin").mkdir(parents=True)
    (source / "empty").mkdir()
    helper = source / "nested" / "bin" / "helper.sh"
    helper.write_text("#!/bin/sh\nprintf 'answer'\n", encoding="utf-8")
    helper.chmod(0o4700)
    program = source / "solve.py"
    program.write_text("answer", encoding="utf-8")
    program.chmod(0o600)
    candidate = Candidate.model_validate_json(pack_candidate("source", source).model_dump_json())
    destination = tmp_path / "private"
    previous = os.umask(0o077)
    try:
        materialize_candidate(candidate, destination)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "nested").stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "nested" / "bin").stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "nested" / "bin" / "helper.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "solve.py").stat().st_mode) == 0o644
    assert not (destination / "empty").exists()
    result = subprocess.run(
        [str(destination / "nested" / "bin" / "helper.sh")],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout == "answer"
    assert pack_candidate("repacked", destination).digest() == candidate.digest()


@pytest.mark.parametrize("existing", ["file", "directory", "symlink"])
def test_materialization_refuses_existing_destination(tmp_path: Path, existing: str) -> None:
    destination = tmp_path / "existing"
    if existing == "file":
        destination.write_bytes(b"original")
    elif existing == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing")
    candidate = Candidate(id="safe", files={"a": "YQ=="})
    with pytest.raises(FileExistsError):
        materialize_candidate(candidate, destination)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped", "nested/../escaped"])
def test_materialize_files_rejects_unsafe_snapshot_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid candidate path"):
        materialize_files(tmp_path / "restored", [(name, b"x", False)])
    assert not (tmp_path / "escaped").exists()


def test_materialize_files_preserves_oversized_research_snapshot(tmp_path: Path) -> None:
    content = b"x" * (10 * 1024 * 1024 + 1)
    destination = tmp_path / "restored"
    materialize_files(destination, [("index.bin", content, False)])
    assert (destination / "index.bin").read_bytes() == content
