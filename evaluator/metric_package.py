import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from evaluator.contracts import Metric, StrictModel
from evaluator.filesystem import candidate_paths

LOCK_NAME = "metric.lock.json"


class PackageFile(StrictModel):
    executable: bool = Field(strict=True)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_checksum(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {name: item for name, item in value.items() if name != "sha256"}
        return value


class PackageLock(StrictModel):
    format_version: Literal[1] = 1
    files: dict[str, PackageFile] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def safe_files(cls, files: dict[str, PackageFile]) -> dict[str, PackageFile]:
        for name in files:
            _validate_relative(name)
            if PurePosixPath(name).name == LOCK_NAME:
                raise ValueError("metric.lock.json is reserved for the package root lock")
        if "metric.json" not in files:
            raise ValueError("metric lock requires metric.json")
        return files


def _validate_relative(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name or name == "." or path.is_absolute() or path.as_posix() != name
        or ".." in path.parts or "\x00" in name or "\\" in name
    ):
        raise ValueError("metric package paths must be normalized and relative")
    if any(part in {".git", ".env"} or part.startswith(".env.") for part in path.parts):
        raise ValueError("metric package contains a restricted path")


def _read_file(path: Path) -> tuple[bytes, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        mode = os.fstat(stream.fileno()).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError("metric package must contain only regular files")
        return stream.read(), mode


def _inventory(directory: Path) -> PackageLock:
    if directory.is_symlink():
        raise ValueError("metric package symlinks are forbidden")
    if not directory.is_dir():
        raise ValueError("metric package must be a directory")
    files = {}
    for path in sorted(candidate_paths(directory)):
        name = path.relative_to(directory).as_posix()
        _validate_relative(name)
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("metric package symlinks are forbidden")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("metric package must contain only regular files")
        if name == LOCK_NAME:
            continue
        files[name] = PackageFile(executable=bool(mode & 0o111))
    return PackageLock(files=files)


def metric_package_provenance(path: Path) -> dict[str, Any] | None:
    if path.is_file():
        return None
    actual = _inventory(path)
    lock_path = path / LOCK_NAME
    if not lock_path.exists():
        raise ValueError("metric package requires metric.lock.json; freeze it before use")
    expected = PackageLock.model_validate_json(_read_file(lock_path)[0])
    if actual != expected:
        raise ValueError("metric package inventory differs from lock")
    return expected.model_dump()


def _expand_command(command: dict[str, Any], directory: Path) -> dict[str, Any]:
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("metric command requires a nonempty string argv")
    expanded = []
    for argument in argv:
        path = PurePosixPath(argument)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("metric package command paths must be relative")
        local = directory / argument
        expanded.append(str(local) if argument and local.exists() else argument)
    executable = expanded[0]
    if not Path(executable).is_absolute():
        if executable in {"python", "python3"}:
            executable = sys.executable
        elif "/" not in executable:
            executable = shutil.which(executable) or ""
        else:
            executable = ""
    if not executable or not os.access(executable, os.X_OK) or not Path(executable).is_file():
        raise ValueError("metric command executable is not available")
    executable = str(Path(executable).absolute())
    expanded[0] = executable
    return {**command, "argv": expanded}


def _load_package(directory: Path) -> Metric:
    directory = directory.absolute()
    configuration = json.loads(_read_file(directory / "metric.json")[0])
    if not isinstance(configuration, dict):
        raise ValueError("metric.json must contain an object")
    for name in ("command", "aggregation"):
        command = configuration.get(name)
        if command is not None:
            if not isinstance(command, dict):
                raise ValueError("metric command must be an object")
            configuration[name] = _expand_command(command, directory)
    return Metric.model_validate(configuration)


def load_metric(path: Path) -> Metric:
    if path.is_file():
        return Metric.model_validate_json(path.read_bytes())
    provenance = metric_package_provenance(path)
    if provenance is None:
        raise ValueError("metric package must be a directory")
    return _load_package(path)


def freeze_metric(source: Path, output: Path) -> dict[str, Any]:
    if source.is_symlink():
        raise ValueError("metric package symlinks are forbidden")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("metric output must not be inside its source directory")
    if (source / LOCK_NAME).exists():
        raise ValueError("metric source already has a lock; verify or copy the frozen package")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"metric output already exists: {output}")
    inventory = _inventory(source)
    _load_package(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metric-", dir=output.parent) as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        for name, item in inventory.files.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data, _ = _read_file(source / name)
            target.write_bytes(data)
            target.chmod(0o755 if item.executable else 0o644)
        if _inventory(staged) != inventory:
            raise ValueError("metric package changed while freezing")
        (staged / LOCK_NAME).write_text(inventory.model_dump_json(indent=2) + "\n")
        load_metric(staged)
        provenance = metric_package_provenance(staged)
        if provenance is None:
            raise ValueError("metric package must be a directory")
        output.mkdir()
        try:
            staged.rename(output)
        except Exception:
            output.rmdir()
            raise
        return provenance
