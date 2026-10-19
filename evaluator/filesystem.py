import os
import stat
from collections.abc import Iterator
from pathlib import Path


def candidate_paths(directory: Path) -> Iterator[Path]:
    pending = [directory]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                yield path
                if stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode):
                    pending.append(path)
