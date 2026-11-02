import base64
import fcntl
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from evaluator.metrics import CommandMetricError
from evaluator.reliability import add_usage, empty_usage
from evaluator.runner import SolveResult
from evaluator.service import Settings, create_app
from evaluator.trace_store import append_call
from evaluator.usage import add_usage as record_usage
from evaluator.usage import create_usage, seal_usage


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps({"id": str(i), "input": f"task-{i}"})
                                   for i in range(3)))
    script = tmp_path / "metric.py"
    script.write_text("print('{\"score\": 4}')\n")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"kind": "command", "version": "v1", "command": {
        "argv": [sys.executable, str(script)], "uses_model": True,
    }}))
    settings = Settings(
        dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
        token="test", runner_image="fixed", release_id="test", finalists=2,
        model="internal", model_base_url="http://internal/v1", retry_delay=0,
        usage_dir=tmp_path / "usage", trace_dir=tmp_path / "traces",
    )
    state = {"fail_answer": "a:task-1", "fail": True, "solves": [], "judges": []}

    def solve(candidate, task):
        answer = f"{candidate.name}:{task}"
        state["solves"].append(answer)
        return SolveResult(answer, {**empty_usage(), "attempts": 1}, 2.0)

    def judge(metric, example, answer, complete, *, model_access, usage, **kwargs):
        state["judges"].append(answer)
        run_id = str(uuid4())
        create_usage(model_access.usage_dir, run_id)
        model_access.run_ids.append(run_id)
        failed = state["fail"] and answer == state["fail_answer"]
        for _ in range(2 if failed else 1):
            measured = None if failed else {
                "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5,
            }
            add_usage(usage, measured)
            record_usage(model_access.usage_dir, run_id, measured)
            assert settings.trace_dir is not None
            append_call(settings.trace_dir, run_id, {
                "name": "model", "span_type": "LLM", "status": "ERROR" if failed else "OK",
                "attributes": {"usage": measured}, "inputs": {}, "outputs": {},
                "start_time_ns": 1, "end_time_ns": 2,
            })
        seal_usage(model_access.usage_dir, run_id)
        if failed:
            raise CommandMetricError("synthetic outage", {
                "model_run_id": run_id, "model_usage_complete": True,
            })
        return {"score": 4.0, "criteria": {}}

    monkeypatch.setattr("evaluator.service.evaluate_answer", judge)
    batch = {"candidates": [{"id": name, "files": {
        "prompt.txt": base64.b64encode(name.encode()).decode(),
    }} for name in ("a", "b")]}

    def start():
        app = create_app(settings, solve=solve)
        api = TestClient(app, headers={"Authorization": "Bearer test"})
        assert api.post("/v1/evaluations", json=batch).status_code == 202
        assert api.get("/v1/evaluations").json() == {"status": "failed"}
        record_path = next(settings.state_dir.glob("*/record.json"))
        return app, api, record_path

    return settings, state, batch, start, solve


def test_resume_reuses_answers_completed_prefix_and_preserves_failed_cost(failed_run):
    settings, state, batch, start, _ = failed_run
    app, api, record_path = start()
    details_path = record_path.parent / "details/a.json"
    before = json.loads(details_path.read_text())
    failed = deepcopy(before["examples"][-1])
    state["fail"] = False
    assert app.state.resume_private("operator authorized retry") == {"status": "complete"}
    after = json.loads(details_path.read_text())
    assert after["examples"][0] == before["examples"][0]
    retried = after["examples"][1]
    assert retried["answer"] == failed["answer"]
    assert retried["resources"]["solver"] == failed["resources"]["solver"]
    assert retried["resources"]["judge"]["usage"]["attempts"] == 3
    assert retried["resources"]["judge"]["usage"]["unreported_attempts"] == 2
    assert (
        retried["resources"]["judge"]["wall_seconds"]
        >= failed["resources"]["judge"]["wall_seconds"]
    )
    assert retried["calls"][:len(failed["calls"])] == failed["calls"]
    assert "failure" not in retried and "diagnostics" not in retried
    assert state["solves"] == [f"{name}:task-{i}" for name in ("a", "b") for i in range(3)]
    assert state["judges"].count("a:task-1") == 2
    record = json.loads(record_path.read_text())
    assert record["resumptions"][0]["reason"] == "operator authorized retry"
    assert record["resumptions"][0]["previous_failure"]["type"] == "CommandMetricError"
    assert list((record_path.parent / "resumptions").glob("*/record.json"))
    assert api.post("/v1/evaluations", json=batch).json() == {"status": "complete"}
    with pytest.raises(ValueError, match="failed"):
        app.state.resume_private("again")


def test_resume_reuses_completed_candidate_without_reaggregation(failed_run, monkeypatch):
    _, state, _, start, _ = failed_run
    state["fail_answer"] = "b:task-1"
    app, _, record_path = start()
    first = record_path.parent / "details/a.json"
    original = first.read_bytes()
    state["fail"] = False
    app.state.resume_private("resume")
    assert first.read_bytes() == original
    assert state["solves"] == [f"{name}:task-{i}" for name in ("a", "b") for i in range(3)]


@pytest.mark.parametrize("damage", ["partial_vote", "missing_trace", "wrong_example", "gap",
                                    "candidate", "provenance", "unknown_status", "usage"])
def test_resume_rejects_unsafe_checkpoint_before_any_work(failed_run, damage):
    settings, state, _, start, _ = failed_run
    app, _, record_path = start()
    path = record_path.parent / "details/a.json"
    document = json.loads(path.read_text())
    failed = document["examples"][-1]
    record = json.loads(record_path.read_text())
    if damage == "partial_vote":
        run_id = failed["resources"]["judge"]["run_id"]
        trace = next((settings.trace_dir / run_id).glob("*.json"))
        value = json.loads(trace.read_text())
        value["status"] = "OK"
        trace.write_text(json.dumps(value))
    elif damage == "missing_trace":
        failed["calls"] = []
    elif damage == "wrong_example":
        failed["example_id"] = "changed"
    elif damage == "gap":
        (path.parent / "b.json").write_text(json.dumps({"candidate_id": "b", "examples": []}))
    elif damage == "candidate":
        (Path(record["candidates"][1]["path"]) / "prompt.txt").write_text("changed")
    elif damage == "provenance":
        record["provenance"]["model"] = "changed"
        record_path.write_text(json.dumps(record))
    elif damage == "unknown_status":
        failed["failure"] = "unknown"
    else:
        failed["resources"]["judge"]["usage"]["attempts"] += 1
    path.write_text(json.dumps(document))
    previous = deepcopy(state)
    with pytest.raises((ValueError, RuntimeError)):
        app.state.resume_private("resume")
    assert state == previous
    assert json.loads(record_path.read_text())["status"] == "failed"


def test_resume_rejects_concurrent_execution_and_public(failed_run):
    settings, _, _, start, solve = failed_run
    app, _, record_path = start()
    with (record_path.parent / "execution.lock").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            app.state.resume_private("resume")
    public = create_app(replace(settings, split="public"), solve=solve)
    with pytest.raises(ValueError, match="private"):
        public.state.resume_private("resume")


def test_repeated_failed_resume_preserves_all_attempts_and_the_seal(failed_run):
    _, state, batch, start, _ = failed_run
    app, api, record_path = start()
    old_record = json.loads(record_path.read_text())
    assert app.state.resume_private("first retry") == {"status": "failed"}
    assert api.post("/v1/evaluations", json=batch).json() == {"status": "failed"}
    state["fail"] = False
    assert app.state.resume_private("second retry") == {"status": "complete"}
    record = json.loads(record_path.read_text())
    assert record["batch_digest"] == old_record["batch_digest"]
    assert record["candidates"] == old_record["candidates"]
    assert record["provenance"] == old_record["provenance"]
    assert [item["status"] for item in record["resumptions"]] == ["failed", "complete"]
    details = json.loads((record_path.parent / "details/a.json").read_text())["examples"][1]
    judge = details["resources"]["judge"]
    assert judge["usage"]["attempts"] == 5
    assert judge["usage"]["unreported_attempts"] == 4
    assert len(judge["run_ids"]) == 3
    assert state["solves"].count("a:task-1") == 1


def test_successful_solver_calls_do_not_block_judge_resume(failed_run):
    _, state, _, start, _ = failed_run
    app, _, record_path = start()
    path = record_path.parent / "details/a.json"
    document = json.loads(path.read_text())
    failed = document["examples"][-1]
    solver = next(call for call in failed["calls"] if call.get("name") == "solver")
    failed["calls"].append({
        "span_type": "LLM", "status": "OK", "parent_id": solver["span_id"], "attributes": {},
    })
    path.write_text(json.dumps(document))
    state["fail"] = False
    assert app.state.resume_private("resume") == {"status": "complete"}


@pytest.mark.parametrize("status", ["complete", "failed", "resume_ready"])
def test_resume_cli_reports_only_status_and_exit_code(monkeypatch, capsys, status):
    from types import SimpleNamespace

    from evaluator import resume

    monkeypatch.setattr(sys, "argv", ["resume", "--reason", "approved technical retry"])
    monkeypatch.setattr(resume.Settings, "from_env", lambda: None)
    monkeypatch.setattr(resume, "create_app", lambda settings: SimpleNamespace(
        state=SimpleNamespace(resume_private=lambda reason, **kwargs: {"status": status}),
    ))
    assert resume.main() == (1 if status == "failed" else 0)
    assert json.loads(capsys.readouterr().out) == {"status": status}


def test_resume_cli_redacts_rejection(monkeypatch, capsys):
    from evaluator import resume

    def rejected():
        raise ValueError("private sensitive diagnostic")

    monkeypatch.setattr(sys, "argv", ["resume", "--reason", "approved technical retry"])
    monkeypatch.setattr(resume.Settings, "from_env", rejected)
    assert resume.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "resume_rejected", "error_type": "ValueError",
    }


@pytest.mark.parametrize("unknown", ["missing_tail", "missing_details", "interrupted"])
def test_resume_rejects_unknown_progress_without_repeating_answers(failed_run, unknown):
    _, state, _, start, _ = failed_run
    app, _, record_path = start()
    path = record_path.parent / "details/a.json"
    document = json.loads(path.read_text())
    if unknown == "missing_tail":
        document["examples"].pop()
        path.write_text(json.dumps(document))
    elif unknown == "missing_details":
        path.unlink()
    else:
        record = json.loads(record_path.read_text())
        record["failure"] = "interrupted"
        record_path.write_text(json.dumps(record))
    previous = deepcopy(state)
    with pytest.raises(ValueError):
        app.state.resume_private("resume")
    assert state == previous


def test_resume_check_validates_without_changing_state_or_calling_models(failed_run):
    settings, state, _, start, _ = failed_run
    app, _, record_path = start()
    before = {str(path): path.read_bytes() for path in settings.state_dir.rglob("*.json")}
    previous = deepcopy(state)
    assert app.state.resume_private("check", check_only=True) == {
        "status": "resume_ready", "completed_examples": 1, "saved_answers": 2,
    }
    assert state == previous
    assert {str(path): path.read_bytes() for path in settings.state_dir.rglob("*.json")} == before
    assert not (record_path.parent / "resumptions").exists()
