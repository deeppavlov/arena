import base64
import hashlib
import json
import math
from pathlib import PurePosixPath

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


class Metric(StrictModel):
    version: str = Field(min_length=1, max_length=100)
    instruction: str = Field(min_length=1, max_length=20000)
    criteria: list[Criterion] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_criteria(self) -> "Metric":
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("criterion IDs must be unique")
        return self


class Example(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    input: str = Field(min_length=1, max_length=500000)
    reference: str = Field(min_length=1, max_length=500000)


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
    candidates: list[Candidate] = Field(min_length=1, max_length=5)

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
