import base64
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator.contracts import Candidate


def validate_candidate_path(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {".", ".."} for part in name.split("/"))
        or "\\" in name
        or "\x00" in name
        or str(path) != name
    ):
        raise ValueError("invalid candidate path")


def candidate_digest(files: Mapping[str, str], executable_files: Sequence[str] = ()) -> str:
    hashes = {
        name: hashlib.sha256(base64.b64decode(data)).hexdigest()
        for name, data in sorted(files.items())
    }
    identity = (
        {"files": hashes, "executable_files": sorted(executable_files)}
        if executable_files else hashes
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def materialize_files(directory: Path, files: Iterable[tuple[str, bytes, bool]]) -> None:
    directory.mkdir(parents=True, mode=0o755)
    directory.chmod(0o755)
    for name, data, executable in files:
        validate_candidate_path(name)
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        for parent in path.parents:
            if parent == directory:
                break
            parent.chmod(0o755)
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o755 if executable else 0o644)


def materialize_candidate(candidate: "Candidate", directory: Path) -> None:
    materialize_files(directory, (
        (name, base64.b64decode(data, validate=True), name in candidate.executable_files)
        for name, data in candidate.files.items()
    ))
