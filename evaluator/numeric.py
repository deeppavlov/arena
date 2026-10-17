import json
import math

MAX_EXACT_SCORE_INTEGER = 2**53 - 1


def validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a finite number")
    if isinstance(value, int) and abs(value) > MAX_EXACT_SCORE_INTEGER:
        raise ValueError("integer score exceeds the exact binary64 range")
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return float(value)


def parse_score(value: str) -> float:
    return validate_score(json.loads(value))
