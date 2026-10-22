import runpy
from pathlib import Path

import pytest

BUILD = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "acceptance/export_acceptance.py"),
)["build_record"]


def source(scenario: str = "retrieval", status: str = "accepted") -> dict:
    return {
        "synthetic_only": True, "configuration_sha256": "a" * 64,
        "settings": {"runner": {"image": "sha256:fixed"}},
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
