import math
from typing import Any

RANKING_POLICY = "score-direction-known-tokens-asc-seconds-asc-id-asc-v2"


def choose_winners(
    scores: list[dict[str, Any]], count: int = 1, direction: str = "maximize",
) -> list[str]:
    if not scores:
        raise ValueError("scores must not be empty")
    if type(count) is not int or count < 1 or count > len(scores):
        raise ValueError("count must be a positive integer no greater than the available scores")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
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
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise ValueError(f"{field} must be finite")
            if field == "solver_seconds" and value < 0:
                raise ValueError("solver_seconds must be nonnegative")
        for field in ("solver_tokens", "solver_unreported_attempts"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
    ranked = sorted(
        scores,
        key=lambda row: (
            -row["score"] if direction == "maximize" else row["score"],
            row["solver_unreported_attempts"] > 0,
            row["solver_tokens"] if row["solver_unreported_attempts"] == 0 else 0,
            row["solver_seconds"],
            row["candidate_id"],
        ),
    )
    return [row["candidate_id"] for row in ranked[:count]]


def choose_winner(scores: list[dict[str, Any]], direction: str = "maximize") -> str:
    return choose_winners(scores, direction=direction)[0]
