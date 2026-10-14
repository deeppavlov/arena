import math
import time
from collections.abc import Callable
from typing import Any

import httpx

MODEL_TIMEOUT = 21600
RUNNER_TIMEOUT = 604800
PREPARE_TIMEOUT = 172800
MODEL_RETRIES = 5
RETRY_DELAY = 30

def validate_timing(value: Any, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return result


def retry_delays(
    retries: int = MODEL_RETRIES, retry_delay: float = RETRY_DELAY
) -> list[float]:
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    delay = min(validate_timing(retry_delay, "retry_delay", allow_zero=True), 300)
    delays = []
    for _ in range(retries):
        delays.append(delay)
        delay = min(delay * 2, 300)
    return delays


def request_timeout(
    timeout: float, retries: int = MODEL_RETRIES, retry_delay: float = RETRY_DELAY
) -> float:
    delays = retry_delays(retries, retry_delay)
    result = (retries + 1) * validate_timing(timeout, "timeout") + sum(delays) + 60
    return validate_timing(result, "request timeout")


def empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "attempts": 0,
        "unreported_attempts": 0,
    }


def parse_usage(value: Any) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(value, dict) or any(
        type(value.get(key)) is not int or value[key] < 0 for key in keys
    ):
        raise ValueError("usage must include nonnegative integer token counts")
    result = {key: value[key] for key in keys}
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
    return result


def add_usage(target: dict[str, int], usage: dict[str, int] | None) -> None:
    if usage is None:
        target["attempts"] += 1
        target["unreported_attempts"] += 1
        return
    parsed = parse_usage(usage)
    for key, value in parsed.items():
        target[key] += value
    target["attempts"] += 1


def retry_call[T](
    operation: Callable[[], T],
    retries: int = MODEL_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> T:
    delays = iter(retry_delays(retries, retry_delay))
    while True:
        try:
            return operation()
        except (httpx.HTTPError, ValueError, TimeoutError, OSError):
            delay = next(delays, None)
            if delay is None:
                raise
            time.sleep(delay)
