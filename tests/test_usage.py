import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest

from evaluator.usage import (
    add_usage,
    begin_request,
    create_usage,
    end_request,
    ensure_open,
    read_usage,
    seal_usage,
)


def counters() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "attempts": 0,
        "unreported_attempts": 0,
    }


def state(**changes: object) -> dict[str, object]:
    return {**counters(), "active_requests": 0, "closed": False, **changes}


def increment_many(arguments: tuple[Path, str]) -> None:
    directory, run_id = arguments
    for _ in range(10):
        add_usage(
            directory, run_id, {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        )


def test_create_and_accumulate_reported_and_unreported_attempts(tmp_path: Path) -> None:
    directory = tmp_path / "usage"
    run_id = str(uuid4())
    create_usage(directory, run_id)
    assert read_usage(directory, run_id) == counters()

    add_usage(directory, run_id, {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
    add_usage(directory, run_id, None)
    add_usage(directory, run_id, {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7})

    assert read_usage(directory, run_id) == {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "attempts": 3,
        "unreported_attempts": 1,
    }
    assert not list(directory.glob("*.tmp"))


def test_create_never_resets_existing_usage(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    add_usage(tmp_path, run_id, None)
    with pytest.raises(FileExistsError):
        create_usage(tmp_path, run_id)
    assert read_usage(tmp_path, run_id) == {**counters(), "attempts": 1, "unreported_attempts": 1}


@pytest.mark.parametrize("operation", ["create", "read", "add"])
@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "../secret",
        "/tmp/secret",
        "not-a-uuid",
        "0" * 32,
        "ABCDEFAB-CDEF-4ABC-8DEF-ABCDEFABCDEF",
    ],
)
def test_rejects_noncanonical_or_unsafe_identifiers(
    tmp_path: Path, operation: str, run_id: str
) -> None:
    directory = tmp_path / "usage"
    with pytest.raises(ValueError):
        if operation == "create":
            create_usage(directory, run_id)
        elif operation == "read":
            read_usage(directory, run_id)
        else:
            add_usage(directory, run_id, None)
    assert not directory.exists()


@pytest.mark.parametrize("existing_directory", [True, False])
def test_read_and_add_never_create_records(tmp_path: Path, existing_directory: bool) -> None:
    directory = tmp_path if existing_directory else tmp_path / "missing"
    run_id = str(uuid4())
    with pytest.raises(FileNotFoundError):
        read_usage(directory, run_id)
    with pytest.raises(FileNotFoundError):
        add_usage(directory, run_id, None)
    assert not (directory / f"{run_id}.json").exists()
    assert not (directory / f"{run_id}.lock").exists()
    if not existing_directory:
        assert not directory.exists()


@pytest.mark.parametrize(
    "record",
    [
        [],
        None,
        {},
        counters(),
        state(extra=0),
        state(attempts=-1),
        state(attempts=True),
        state(attempts=1.0),
        state(prompt_tokens="0"),
        state(total_tokens=1),
        state(completion_tokens=-1),
        state(unreported_attempts=1),
        state(active_requests=-1),
        state(active_requests=True),
        state(active_requests=1.0),
        state(closed=0),
        state(closed="false"),
    ],
)
def test_read_and_add_reject_malformed_records(tmp_path: Path, record: object) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    path = tmp_path / f"{run_id}.json"
    content = json.dumps(record)
    path.write_text(content)
    with pytest.raises(ValueError):
        read_usage(tmp_path, run_id)
    with pytest.raises(ValueError):
        add_usage(tmp_path, run_id, None)
    assert path.read_text() == content


def test_invalid_json_is_not_overwritten(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    path = tmp_path / f"{run_id}.json"
    path.write_text("{")
    with pytest.raises(ValueError):
        read_usage(tmp_path, run_id)
    with pytest.raises(ValueError):
        add_usage(tmp_path, run_id, None)
    assert path.read_text() == "{"


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4},
        {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
    ],
)
def test_malformed_response_usage_leaves_record_unchanged(
    tmp_path: Path, usage: dict[str, int]
) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    with pytest.raises(ValueError):
        add_usage(tmp_path, run_id, usage)
    assert read_usage(tmp_path, run_id) == counters()


def test_failed_save_preserves_previous_record_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    add_usage(tmp_path, run_id, None)
    previous = read_usage(tmp_path, run_id)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("Disk error")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Disk error"):
        add_usage(tmp_path, run_id, None)
    assert read_usage(tmp_path, run_id) == previous
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("executor_type", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_concurrent_writers_do_not_lose_usage(
    tmp_path: Path, executor_type: type[ThreadPoolExecutor] | type[ProcessPoolExecutor]
) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    with executor_type(max_workers=4) as executor:
        list(executor.map(increment_many, [(tmp_path, run_id)] * 8))
    assert read_usage(tmp_path, run_id) == {
        "prompt_tokens": 160,
        "completion_tokens": 240,
        "total_tokens": 400,
        "attempts": 80,
        "unreported_attempts": 0,
    }


def test_sealed_usage_rejects_new_requests(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    assert seal_usage(tmp_path, run_id) == counters()
    with pytest.raises(ValueError, match="closed"):
        begin_request(tmp_path, run_id)
    assert seal_usage(tmp_path, run_id) == counters()
    assert read_usage(tmp_path, run_id) == counters()


def test_successfully_sealed_usage_cannot_be_modified(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    assert seal_usage(tmp_path, run_id) == counters()
    with pytest.raises(ValueError, match="closed"):
        add_usage(tmp_path, run_id, None)
    assert read_usage(tmp_path, run_id) == counters()


def test_sealing_active_request_closes_run_before_raising(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    begin_request(tmp_path, run_id)
    begin_request(tmp_path, run_id)
    with pytest.raises(ValueError, match="active"):
        seal_usage(tmp_path, run_id)
    with pytest.raises(ValueError, match="closed"):
        begin_request(tmp_path, run_id)

    add_usage(tmp_path, run_id, {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
    end_request(tmp_path, run_id)
    with pytest.raises(ValueError, match="active"):
        seal_usage(tmp_path, run_id)
    end_request(tmp_path, run_id)
    expected = {
        **counters(), "prompt_tokens": 2, "completion_tokens": 3,
        "total_tokens": 5, "attempts": 1,
    }
    assert seal_usage(tmp_path, run_id) == expected
    with pytest.raises(ValueError, match="closed"):
        begin_request(tmp_path, run_id)


def test_end_request_cannot_make_active_count_negative(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    with pytest.raises(ValueError, match="active"):
        end_request(tmp_path, run_id)
    assert seal_usage(tmp_path, run_id) == counters()


@pytest.mark.parametrize("operation", [begin_request, end_request, seal_usage, ensure_open])
def test_lifecycle_never_creates_missing_run(tmp_path: Path, operation: object) -> None:
    directory = tmp_path / "missing"
    assert callable(operation)
    with pytest.raises(FileNotFoundError):
        operation(directory, str(uuid4()))
    assert not directory.exists()


def test_ensure_open_does_not_change_open_run(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    path = tmp_path / f"{run_id}.json"
    before = path.read_bytes()
    ensure_open(tmp_path, run_id)
    assert path.read_bytes() == before


@pytest.mark.parametrize("active", [True, False])
def test_ensure_open_rejects_closed_run_with_or_without_active_requests(
    tmp_path: Path, active: bool
) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path, run_id)
    if active:
        begin_request(tmp_path, run_id)
        with pytest.raises(ValueError, match="active"):
            seal_usage(tmp_path, run_id)
    else:
        seal_usage(tmp_path, run_id)
    with pytest.raises(ValueError, match="closed"):
        ensure_open(tmp_path, run_id)


def test_seal_racing_begin_is_atomic(tmp_path: Path) -> None:
    for _ in range(20):
        run_id = str(uuid4())
        create_usage(tmp_path, run_id)
        barrier = Barrier(2)

        def begin(request_id: str, synchronization: Barrier) -> bool:
            synchronization.wait()
            try:
                begin_request(tmp_path, request_id)
            except ValueError:
                return False
            return True

        def seal(request_id: str, synchronization: Barrier) -> bool:
            synchronization.wait()
            try:
                seal_usage(tmp_path, request_id)
            except ValueError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            beginning = executor.submit(begin, run_id, barrier)
            sealing = executor.submit(seal, run_id, barrier)
            started, sealed = beginning.result(), sealing.result()
        assert started != sealed
        with pytest.raises(ValueError, match="closed"):
            begin_request(tmp_path, run_id)
        if started:
            end_request(tmp_path, run_id)
        assert seal_usage(tmp_path, run_id) == counters()
