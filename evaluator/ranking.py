import math
from typing import Any

RANKING_POLICY = "score-desc-known-tokens-asc-seconds-asc-id-asc-v1"


def choose_winner(scores: list[dict[str, Any]]) -> str:
    if not scores:
        raise ValueError("scores must not be empty")
    identifiers: set[str] = set()
    for row in scores:
        for field in (
            "candidate_id",
            "score",
            "solver_tokens",
            "solver_seconds",
            "solver_unreported_attempts",
        ):
            if field not in row:
                raise ValueError(f"missing {field}")
        identifier = row["candidate_id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("candidate_id must be a nonempty string")
        if identifier in identifiers:
            raise ValueError("candidate_id must be unique")
        identifiers.add(identifier)
        for field in ("score", "solver_seconds"):
            value = row[field]
            if (
                type(value) not in (int, float)
                or value < 0
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise ValueError(f"{field} must be finite and nonnegative")
        for field in ("solver_tokens", "solver_unreported_attempts"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
    winner = min(
        scores,
        key=lambda row: (
            -row["score"],
            row["solver_unreported_attempts"] > 0,
            row["solver_tokens"] if row["solver_unreported_attempts"] == 0 else 0,
            row["solver_seconds"],
            row["candidate_id"],
        ),
    )
    return winner["candidate_id"]
