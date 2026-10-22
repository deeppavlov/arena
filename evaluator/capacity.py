import fcntl
import os
import stat
from pathlib import Path
from typing import BinaryIO


class CapacityError(RuntimeError):
    pass


def _validate_owner(descriptor: int, directory: bool) -> None:
    metadata = os.fstat(descriptor)
    valid_type = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not valid_type or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise ValueError("Capacity storage must be operator-owned and not writable by others")


def acquire_slot(directory: Path | None, slots: int) -> BinaryIO | None:
    if type(slots) is not int or slots <= 0:
        raise ValueError("Capacity slots must be a positive integer")
    if directory is None:
        return None
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _validate_owner(directory_fd, directory=True)
        for index in range(slots):
            descriptor = os.open(
                f"slot-{index}.lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            lease = os.fdopen(descriptor, "r+b")
            try:
                _validate_owner(lease.fileno(), directory=False)
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            except BaseException:
                lease.close()
                raise
            return lease
    finally:
        os.close(directory_fd)
    raise CapacityError("Evaluation capacity is busy")
