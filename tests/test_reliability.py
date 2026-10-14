import math

import httpx
import pytest

from evaluator import reliability


def test_retry_schedule_and_timeout_include_all_attempts():
    assert reliability.retry_delays() == [30, 60, 120, 240, 300]
    assert reliability.request_timeout(100) == 1410
    assert reliability.retry_delays(0) == []
    assert reliability.request_timeout(100, retries=0) == 160


@pytest.mark.parametrize("retries", [-1, 1.5, True, "5"])
def test_invalid_retry_count(retries):
    with pytest.raises(ValueError):
        reliability.retry_delays(retries)


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, None, "bad"])
def test_invalid_timing(value):
    with pytest.raises(ValueError):
        reliability.validate_timing(value, "timeout")


def test_timing_accepts_env_values_and_zero_delay():
    assert reliability.validate_timing("21600", "timeout") == 21600
    assert reliability.validate_timing(0, "delay", allow_zero=True) == 0
    assert reliability.retry_delays(3, 0) == [0, 0, 0]


@pytest.mark.parametrize("delay", [-1, math.inf, math.nan, False])
def test_invalid_retry_delay(delay):
    with pytest.raises(ValueError):
        reliability.retry_delays(retry_delay=delay)


def test_usage_counts_all_attempts_and_missing_provider_reports():
    usage = reliability.empty_usage()
    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "attempts": 0,
        "unreported_attempts": 0,
    }
    reported = reliability.parse_usage({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 4},
    })
    reliability.add_usage(usage, reported)
    reliability.add_usage(usage, None)
    reliability.add_usage(usage, reported)
    assert usage == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "attempts": 3,
        "unreported_attempts": 1,
    }


@pytest.mark.parametrize("value", [
    None,
    {},
    {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4},
    {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1},
    {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
    {"prompt_tokens": 1.0, "completion_tokens": 2, "total_tokens": 3},
])
def test_invalid_provider_usage(value):
    with pytest.raises(ValueError):
        reliability.parse_usage(value)


@pytest.mark.parametrize("error", [
    ValueError("malformed response"),
    httpx.ReadTimeout("slow response"),
    TimeoutError("slow operation"),
    OSError("connection interrupted"),
])
def test_retry_call_allows_five_retries(monkeypatch, error):
    calls = []
    sleeps = []
    monkeypatch.setattr(reliability.time, "sleep", sleeps.append)

    def operation():
        calls.append(None)
        if len(calls) <= 5:
            raise error
        return "answer"

    assert reliability.retry_call(operation) == "answer"
    assert len(calls) == 6
    assert sleeps == [30, 60, 120, 240, 300]


def test_retry_call_reraises_final_error_without_extra_sleep(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(reliability.time, "sleep", sleeps.append)

    def operation():
        calls.append(None)
        raise ValueError(str(len(calls)))

    with pytest.raises(ValueError, match="^6$"):
        reliability.retry_call(operation)
    assert len(calls) == 6
    assert len(sleeps) == 5


def test_retry_call_does_not_hide_programming_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(reliability.time, "sleep", sleeps.append)

    def operation():
        raise TypeError("programming error")

    with pytest.raises(TypeError, match="programming error"):
        reliability.retry_call(operation)
    assert sleeps == []
