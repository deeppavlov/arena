from pathlib import Path
from uuid import uuid4

import pytest

from evaluator.trace_store import append_call, read_calls


def test_trace_files_redact_and_preserve_sorted_attempts(tmp_path: Path) -> None:
    run_id = str(uuid4())
    append_call(tmp_path, run_id, {"name": "second", "start_time_ns": 20, "api_key": "hidden"})
    append_call(tmp_path, run_id, {"name": "first", "start_time_ns": 10})
    assert [call["name"] for call in read_calls(tmp_path, run_id)] == ["first", "second"]
    files = list((tmp_path / run_id).glob("*.json"))
    assert len(files) == 2
    assert all("hidden" not in path.read_text() for path in files)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)


@pytest.mark.parametrize("run_id", ["../outside", "abc", "", "/tmp/outside"])
def test_rejects_invalid_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError):
        append_call(tmp_path, run_id, {})
    with pytest.raises(ValueError):
        read_calls(tmp_path, run_id)
    assert list(tmp_path.iterdir()) == []


def test_no_calls_is_empty(tmp_path: Path) -> None:
    assert read_calls(tmp_path, str(uuid4())) == []


def test_redacts_known_secrets_even_in_unstructured_response(tmp_path: Path) -> None:
    run_id = str(uuid4())
    append_call(
        tmp_path, run_id, {"secret-value": "I echoed secret-value"}, secrets=("secret-value",),
    )
    assert read_calls(tmp_path, run_id) == [{"[REDACTED]": "I echoed [REDACTED]"}]


def test_call_iterator_loads_only_the_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluator.trace_store import call_paths, iter_calls

    run_id = str(uuid4())
    append_call(tmp_path, run_id, {"name": "one"})
    append_call(tmp_path, run_id, {"name": "two"})
    paths = call_paths(tmp_path, run_id)
    original = Path.read_text
    loaded = []

    def read(path: Path, *args: object, **kwargs: object) -> str:
        loaded.append(path)
        return original(path, encoding="utf-8")

    monkeypatch.setattr(Path, "read_text", read)
    calls = iter_calls(tmp_path, run_id)
    assert loaded == []
    next(calls)
    assert loaded == [paths[0]]
    next(calls)
    assert loaded == paths
