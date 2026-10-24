import base64
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluator.candidates import candidate_digest, validate_candidate_path


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
    timeout_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)
    uses_model: bool = Field(default=False, strict=True)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_sources(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {name: item for name, item in value.items() if name != "source_sha256"}
        return value

    @model_validator(mode="after")
    def safe_executable(self) -> "CommandMetric":
        if any("\x00" in argument for argument in self.argv):
            raise ValueError("command arguments must not contain null bytes")
        if not PurePosixPath(self.argv[0]).is_absolute():
            raise ValueError("command executable must be an absolute path")
        return self


class Metric(StrictModel):
    version: str = Field(min_length=1, max_length=100)
    kind: Literal["llm", "command"] = "llm"
    direction: Literal["maximize", "minimize"] = "maximize"
    instruction: str = Field(default="", max_length=20000)
    criteria: list[Criterion] = Field(default_factory=list, max_length=30)
    votes: int = Field(default=3, gt=0, strict=True)
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
        elif self.votes != 3:
            raise ValueError("votes can only configure the built-in LLM metric")
        return self


class Example(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    input: str = Field(max_length=500000)
    reference: str | None = Field(default=None, max_length=500000)


class Candidate(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    files: dict[str, str] = Field(min_length=1, max_length=1000)
    executable_files: list[str] = Field(default_factory=list, max_length=1000)

    @field_validator("files")
    @classmethod
    def safe_files(cls, files: dict[str, str]) -> dict[str, str]:
        total = 0
        for name, value in files.items():
            validate_candidate_path(name)
            if len(name) > 240 or any(
                len(part.encode("utf-8")) > 255 for part in PurePosixPath(name).parts
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

    @field_validator("executable_files")
    @classmethod
    def canonical_executable_files(cls, names: list[str]) -> list[str]:
        if len(set(names)) != len(names):
            raise ValueError("executable files must be unique")
        return sorted(names)

    @model_validator(mode="after")
    def existing_executable_files(self) -> "Candidate":
        if any(name not in self.files for name in self.executable_files):
            raise ValueError("executable files must exist in candidate files")
        return self

    def digest(self) -> str:
        return candidate_digest(self.files, self.executable_files)


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
