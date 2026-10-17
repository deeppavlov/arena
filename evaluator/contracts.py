import base64
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criterion(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    question: str = Field(min_length=1, max_length=10000)
    weight: float = Field(gt=0)

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight must be finite")
        return value


class CommandMetric(StrictModel):
    argv: list[str] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    timeout_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def pinned_executable(self) -> "CommandMetric":
        if any("\x00" in argument for argument in self.argv):
            raise ValueError("command arguments must not contain null bytes")
        if not PurePosixPath(self.argv[0]).is_absolute():
            raise ValueError("command executable must be an absolute path")
        for name, digest in self.source_sha256.items():
            if not PurePosixPath(name).is_absolute() or "\x00" in name:
                raise ValueError("command source paths must be absolute")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("command sources require lowercase SHA-256 hashes")
        if self.argv[0] not in self.source_sha256:
            raise ValueError("command executable must be pinned in source_sha256")
        return self


class Metric(StrictModel):
    version: str = Field(min_length=1, max_length=100)
    kind: Literal["llm", "command"] = "llm"
    direction: Literal["maximize", "minimize"] = "maximize"
    instruction: str = Field(default="", max_length=20000)
    criteria: list[Criterion] = Field(default_factory=list, max_length=30)
    command: CommandMetric | None = None
    aggregation: CommandMetric | None = None

    @model_validator(mode="after")
    def valid_profile(self) -> "Metric":
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("criterion IDs must be unique")
        if self.kind == "llm":
            if not self.instruction.strip() or not self.criteria:
                raise ValueError("LLM metric requires instruction and criteria")
            if self.direction != "maximize" and self.aggregation is None:
                raise ValueError("built-in LLM quality scores must maximize")
            if self.command is not None:
                raise ValueError("LLM metric cannot configure a command")
        elif self.command is None or self.criteria:
            raise ValueError("command metric requires a command and no LLM criteria")
        return self


class Example(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    input: str = Field(max_length=500000)
    reference: str | None = Field(default=None, max_length=500000)


class Candidate(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    files: dict[str, str] = Field(min_length=1, max_length=1000)

    @field_validator("files")
    @classmethod
    def safe_files(cls, files: dict[str, str]) -> dict[str, str]:
        total = 0
        for name, value in files.items():
            path = PurePosixPath(name)
            if (
                not name
                or len(name) > 240
                or path.is_absolute()
                or any(part in {".", ".."} for part in name.split("/"))
                or "\\" in name
                or "\x00" in name
                or str(path) != name
            ):
                raise ValueError("invalid candidate path")
            try:
                data = base64.b64decode(value, validate=True)
            except ValueError as error:
                raise ValueError("candidate files must be base64") from error
            total += len(data)
            if total > 10 * 1024 * 1024:
                raise ValueError("candidate exceeds 10 MiB")
        paths = set(files)
        if any(str(parent) in paths for name in paths for parent in PurePosixPath(name).parents):
            raise ValueError("candidate file conflicts with directory")
        return files

    def digest(self) -> str:
        hashes = {
            name: hashlib.sha256(base64.b64decode(data)).hexdigest()
            for name, data in sorted(self.files.items())
        }
        return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


class Submission(StrictModel):
    candidates: list[Candidate] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_candidates(self) -> "Submission":
        if len({item.id for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate IDs must be unique")
        if len({item.digest() for item in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate candidate content")
        return self

    def digest(self) -> str:
        values = [(candidate.id, candidate.digest()) for candidate in self.candidates]
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class Vote(StrictModel):
    score: int = Field(ge=1, le=5, strict=True)
    reason: str = Field(min_length=1, max_length=20000)
