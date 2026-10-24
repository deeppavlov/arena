import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

EXPORT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "acceptance/export_acceptance.py"),
)
BUILD = EXPORT["build_record"]


def source(scenario: str = "retrieval", status: str = "accepted") -> dict:
    return {
        "synthetic_only": True,
        "settings": {"runner": {"image": "hermes:fixed"}},
        "scenarios": {scenario: {
            "status": status, "started_at_unix": 100, "finished_at_unix": 110,
            "wall_seconds": 10,
            "result": {"answer": "synthetic", "usage": {
                "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
                "attempts": 1, "unreported_attempts": 0,
            }, "wall_seconds": 8, "run_id": "synthetic-uuid"},
        }},
    }


def test_solver_costs_are_separate_and_no_score_is_invented():
    record = BUILD(source(), "retrieval", [])
    assert record["status"] == "FINISHED"
    assert record["metrics"]["solver_tokens"] == 5
    assert record["metrics"]["judge_tokens"] == 0
    assert record["metrics"]["solver_seconds"] == 8
    assert "score" not in record["metrics"]
    assert record["params"]["candidate_id"] == "sc71-synthetic-acceptance"
    assert record["params"]["configuration"] == source()["settings"]


def test_judge_keeps_actual_rejected_score_and_vote_attributes():
    value = source("judge", "rejected")
    value["scenarios"]["judge"]["result"]["evaluation"] = {"score": 2.5, "wall_seconds": 9}
    calls = [{
        "name": "model", "start_time_ns": 101000000000, "end_time_ns": 102000000000,
        "attributes": {"caller_metadata": {"criterion": "accuracy", "vote": 2}},
    }]
    record = BUILD(value, "judge", calls)
    assert record["status"] == "FAILED"
    assert record["metrics"]["score"] == 2.5
    assert record["metrics"]["judge_tokens"] == 5
    assert record["metrics"]["solver_tokens"] == 0
    assert record["calls"][1]["attributes"]["criterion"] == "accuracy"
    assert record["calls"][1]["attributes"]["vote"] == 2


def test_failed_unknown_usage_stays_unknown():
    value = source(status="failed")
    del value["scenarios"]["retrieval"]["result"]
    record = BUILD(value, "retrieval", [])
    assert record["metrics"]["solver_tokens"] is None
    assert record["metrics"]["solver_unreported_attempts"] is None
    assert record["status"] == "FAILED"


@pytest.mark.parametrize("change", ["real", "running", "unknown"])
def test_rejects_non_synthetic_or_unfinished_input(change: str):
    value = source()
    scenario = "retrieval"
    if change == "real":
        value["synthetic_only"] = False
    elif change == "running":
        value["scenarios"][scenario]["status"] = "running"
    else:
        scenario = "unknown"
    with pytest.raises(ValueError):
        BUILD(value, scenario, [])


@pytest.mark.parametrize("changed", [True, False])
def test_saved_export_compares_readable_configuration(tmp_path, monkeypatch, changed):
    from evaluator import tracking

    value = source()
    value["settings"]["api_key"] = "sensitive-value"
    record = BUILD(value, "retrieval", [])
    assert record["params"]["configuration"]["api_key"] == "[REDACTED]"
    output = tmp_path / "output"
    records = output / "records"
    records.mkdir(parents=True)
    (records / "retrieval.json").write_text(json.dumps(record))
    if changed:
        value["settings"]["runner"]["image"] = "hermes:changed"
    state = tmp_path / "state.json"
    state.write_text(json.dumps(value))
    export_record = Mock(return_value={"status": "disabled"})
    monkeypatch.setattr(tracking, "export_record", export_record)
    monkeypatch.setattr(EXPORT["sys"], "argv", [
        "export_acceptance.py", "--state", str(state), "--output", str(output),
        "--scenario", "retrieval",
    ])
    if changed:
        with pytest.raises(ValueError, match="configuration"):
            EXPORT["main"]()
        export_record.assert_not_called()
    else:
        EXPORT["main"]()
        assert export_record.call_count == 2


@pytest.mark.parametrize("changed", [True, False])
def test_legacy_export_reuses_record_without_rewriting_checksums(tmp_path, monkeypatch, changed):
    from evaluator import tracking

    value = source()
    value["settings"]["metric_lock_sha256"] = "unusable"
    record = BUILD(value, "retrieval", [])
    assert "metric_lock_sha256" not in record["params"]["configuration"]
    del record["params"]["configuration"]
    record["params"]["configuration_sha256"] = "unusable"
    if changed:
        value["settings"]["runner"]["image"] = "hermes:changed"
    output = tmp_path / "output"
    records = output / "records"
    records.mkdir(parents=True)
    path = records / "retrieval.json"
    original = json.dumps(record)
    path.write_text(original)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(value))
    export_record = Mock(return_value={"status": "disabled"})
    monkeypatch.setattr(tracking, "export_record", export_record)
    monkeypatch.setattr(EXPORT["sys"], "argv", [
        "export_acceptance.py", "--state", str(state), "--output", str(output),
        "--scenario", "retrieval",
    ])
    if changed:
        with pytest.raises(ValueError, match="configuration"):
            EXPORT["main"]()
        export_record.assert_not_called()
    else:
        EXPORT["main"]()
        assert export_record.call_count == 2
        assert all(call.args[0] == path for call in export_record.call_args_list)
    assert path.read_text() == original
