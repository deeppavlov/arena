import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluator.capacity import CapacityError, acquire_slot


def test_capacity_is_unlimited_without_directory():
    assert acquire_slot(None, 1) is None


@pytest.mark.parametrize("slots", [0, -1, True, 1.5])
def test_capacity_requires_positive_integer_slots(tmp_path: Path, slots):
    with pytest.raises(ValueError, match="positive integer"):
        acquire_slot(tmp_path, slots)


def test_capacity_rejects_full_and_releases_on_close(tmp_path: Path):
    first = acquire_slot(tmp_path, 2)
    second = acquire_slot(tmp_path, 2)
    assert first is not None and second is not None
    try:
        with pytest.raises(CapacityError):
            acquire_slot(tmp_path, 2)
        first.close()
        replacement = acquire_slot(tmp_path, 2)
        assert replacement is not None
        replacement.close()
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("crash", [False, True])
def test_capacity_is_shared_between_processes_and_released(tmp_path: Path, crash: bool):
    program = """
import sys
from pathlib import Path
from evaluator.capacity import acquire_slot
lease = acquire_slot(Path(sys.argv[1]), 1)
print('ready', flush=True)
sys.stdin.readline()
lease.close()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        assert process.stdout.readline() == "ready\n"
        with pytest.raises(CapacityError):
            acquire_slot(tmp_path, 1)
        if crash:
            process.kill()
        else:
            process.stdin.write("release\n")
            process.stdin.flush()
        process.wait(timeout=10)
        replacement = acquire_slot(tmp_path, 1)
        assert replacement is not None
        replacement.close()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_capacity_requires_precreated_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        acquire_slot(tmp_path / "missing", 1)


def test_capacity_rejects_writable_directory(tmp_path: Path):
    tmp_path.chmod(0o777)
    with pytest.raises(ValueError, match="operator-owned"):
        acquire_slot(tmp_path, 1)


def test_capacity_rejects_symlink_directory(tmp_path: Path):
    directory = tmp_path / "capacity"
    directory.mkdir()
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(OSError):
        acquire_slot(link, 1)


def test_capacity_rejects_symlink_slot(tmp_path: Path):
    target = tmp_path / "unrelated"
    target.write_bytes(b"unchanged")
    (tmp_path / "slot-0.lock").symlink_to(target)
    with pytest.raises(OSError):
        acquire_slot(tmp_path, 1)
    assert target.read_bytes() == b"unchanged"


def test_capacity_rejects_writable_slot(tmp_path: Path):
    slot = tmp_path / "slot-0.lock"
    slot.touch(mode=0o600)
    slot.chmod(0o666)
    with pytest.raises(ValueError, match="operator-owned"):
        acquire_slot(tmp_path, 1)


def test_capacity_rejects_foreign_owner(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: tmp_path.stat().st_uid + 1)
    with pytest.raises(ValueError, match="operator-owned"):
        acquire_slot(tmp_path, 1)
