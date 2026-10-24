import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PACKAGE = Path(__file__).parents[1] / "acceptance"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


@pytest.fixture
def acceptance():
    return SimpleNamespace(**runpy.run_path(str(PACKAGE / "run.py")))


def calls(names: list[str]) -> list[dict]:
    result = []
    for number, name in enumerate(names):
        identifier = f"call-{number}"
        result.append({
            "status": "OK", "inputs": {
                "model": MODEL, "reasoning_effort": "max", "messages": [],
            }, "outputs": {"choices": [{"message": {"tool_calls": [{
                "id": identifier, "function": {"name": name, "arguments": "{}"},
            }]}}]},
        })
        result.append({
            "status": "OK", "inputs": {
                "model": MODEL, "reasoning_effort": "max", "messages": [{
                    "role": "tool", "tool_call_id": identifier, "content": json.dumps({
                        "results": [{"status": "completed", "summary": "КОД-СЕВЕР-731"}],
                    }, ensure_ascii=False),
                }],
            }, "outputs": {"choices": [{"message": {"content": "КОД-СЕВЕР-731"}}]},
        })
    return result


@pytest.mark.parametrize("explicit_role", [True, False])
def test_solver_requires_trusted_completed_retrieval_and_nested_delegation(
    acceptance, explicit_role,
):
    trace = calls(["arena_search", "delegate_task", "delegate_task", "arena_search"])
    root_goal = "SC71_ACCEPT_CHILD\nDelegate again to find the code."
    leaf_goal = "SC71_ACCEPT_GRANDCHILD\nFind the code."
    trace[2]["outputs"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        json.dumps({"goal": root_goal} | ({"role": "orchestrator"} if explicit_role else {}))
    )
    trace[4]["inputs"]["messages"] = [
        {"role": "system", "content": (
            "NOTE: You are at depth 1. The delegation tree is capped at max_spawn_depth=3."
        )},
        {"role": "user", "content": root_goal},
    ]
    trace[4]["inputs"]["tools"] = [{"function": {"name": "delegate_task"}}]
    trace[4]["outputs"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        json.dumps({"goal": leaf_goal})
    )
    trace[6]["inputs"]["messages"] = [{"role": "user", "content": leaf_goal}]
    evidence = acceptance.verify_solver("delegation", "КОД-СЕВЕР-731", trace)
    assert evidence["completed_tool_calls"] == {"arena_search": 2, "delegate_task": 2}
    with pytest.raises(ValueError, match="delegation"):
        acceptance.verify_solver("delegation", "КОД-СЕВЕР-731", calls(["arena_search"]))
    with pytest.raises(ValueError, match="retrieval"):
        acceptance.verify_solver("retrieval", "КОД-СЕВЕР-731", trace[:1])
    with pytest.raises(ValueError, match="answer"):
        acceptance.verify_solver("retrieval", "другой код", trace)
    with pytest.raises(ValueError, match="nested"):
        acceptance.verify_solver("delegation", "КОД-СЕВЕР-731", calls([
            "arena_search", "delegate_task", "delegate_task",
        ]))
    with pytest.raises(ValueError, match="root retrieval"):
        acceptance.verify_solver("delegation", "КОД-СЕВЕР-731", trace[2:])
    trace[5]["inputs"]["messages"][0]["content"] = json.dumps({
        "results": [{"status": "failed", "summary": "КОД-СЕВЕР-731"}],
    }, ensure_ascii=False)
    with pytest.raises(ValueError, match="delegation"):
        acceptance.verify_solver("delegation", "КОД-СЕВЕР-731", trace)


def test_traces_must_cover_all_attempts_and_known_tokens(acceptance):
    trace = [
        {"attributes": {"usage": {"prompt_tokens": 5, "completion_tokens": 3,
                                   "total_tokens": 8}}},
        {"attributes": {"usage": None}},
    ]
    usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
             "attempts": 2, "unreported_attempts": 1}
    assert acceptance.verify_usage(usage, trace) == {"token_accounting_complete": False}
    with pytest.raises(ValueError, match="trace"):
        acceptance.verify_usage(usage, trace[:1])
    with pytest.raises(ValueError, match="trace"):
        acceptance.verify_usage(dict(usage, total_tokens=10, completion_tokens=5), trace)


def test_tool_ids_normalized_by_hermes_keep_the_original_completed_call(acceptance):
    trace = calls(["arena_search"])
    original = trace[0]["outputs"]["choices"][0]["message"]["tool_calls"][0]
    originals: list[dict] = [dict(original, id=f"chatcmpl-tool-{name}") for name in ("one", "two")]
    for tool in originals:
        tool["function"] = {"name": "arena_search", "arguments": '{"query": "Северная комета"}'}
    trace[0]["outputs"]["choices"][0]["message"]["tool_calls"] = originals
    replayed: list[dict] = [
        dict(tool, id="call_" + hashlib.sha256(tool["id"].encode()).hexdigest()[:12])
        for tool in originals
    ]
    for tool in replayed:
        tool["function"] = {"name": "arena_search", "arguments": json.dumps(
            {"query": "Северная комета"}, ensure_ascii=True, separators=(",", ":"),
        )}
    content = trace[1]["inputs"]["messages"][0]["content"]
    trace[1]["inputs"]["messages"] = [{"role": "assistant", "tool_calls": replayed}] + [
        {"role": "tool", "tool_call_id": tool["id"], "content": content} for tool in replayed
    ]
    result = acceptance.verify_solver("retrieval", "КОД-СЕВЕР-731", trace)
    assert result["completed_tool_calls"] == {"arena_search": 2}
    trace[1]["inputs"]["messages"][0]["tool_calls"][0]["function"] = {
        "name": "arena_search", "arguments": '{"query":"different"}',
    }
    with pytest.raises(ValueError, match="unmatched"):
        acceptance.verify_solver("retrieval", "КОД-СЕВЕР-731", trace)


@pytest.mark.parametrize("identifiers", [
    ["chatcmpl-tool-one"], ["chatcmpl-tool-one", "other-provider-id"],
])
def test_singleton_and_mixed_tool_batches_cannot_claim_normalized_aliases(acceptance, identifiers):
    trace = calls(["arena_search"])
    function = trace[0]["outputs"]["choices"][0]["message"]["tool_calls"][0]["function"]
    trace[0]["outputs"]["choices"][0]["message"]["tool_calls"] = [
        {"id": identifier, "function": function} for identifier in identifiers
    ]
    alias = "call_" + hashlib.sha256(identifiers[0].encode()).hexdigest()[:12]
    trace[1]["inputs"]["messages"].insert(0, {
        "role": "assistant", "tool_calls": [{"id": alias, "function": function}],
    })
    with pytest.raises(ValueError, match="unmatched"):
        acceptance.verify_solver("retrieval", "КОД-СЕВЕР-731", trace)


@pytest.mark.parametrize("field,value", [("model", "other"), ("reasoning_effort", "high")])
def test_every_trusted_attempt_uses_fixed_model_parameters(acceptance, field, value):
    trace = calls(["arena_search"])
    trace.append({"status": "ERROR", "inputs": dict(trace[0]["inputs"])})
    trace[-1]["inputs"][field] = value
    with pytest.raises(ValueError, match="fixed"):
        acceptance.verify_solver("retrieval", "КОД-СЕВЕР-731", trace)


def test_judge_requires_nine_valid_votes_and_fixed_weighted_score(acceptance):
    scores = {"completeness": [5, 5, 4], "accuracy": [4, 4, 4], "groundedness": [5, 5, 5]}
    trace = [{
        "status": "OK", "inputs": {"model": MODEL, "reasoning_effort": "max"},
        "outputs": {"choices": [{"message": {
            "content": json.dumps({"score": score, "reason": "Соответствует эталону"}),
        }}]},
    } for votes in scores.values() for score in votes]
    result = {"score": 4.533333333333333, "statistics": {"judge_scores": scores}}
    assert acceptance.verify_judge(result, trace)["valid_votes"] == 9
    with pytest.raises(ValueError, match="nine"):
        acceptance.verify_judge(result, trace[:-1])
    with pytest.raises(ValueError, match="weighted"):
        acceptance.verify_judge(dict(result, score=5), trace)
    trace[0]["outputs"]["choices"][0]["message"]["content"] = '{"score": true,"reason":"x"}'
    with pytest.raises(ValueError, match="vote"):
        acceptance.verify_judge(result, trace)


@pytest.mark.parametrize("accepted", [True, False])
def test_recorded_valid_result_is_never_executed_again(acceptance, tmp_path, accepted):
    state = {"scenarios": {}}
    operation = Mock(return_value={"answer": "fixed"})
    verify = Mock(return_value={"ok": True})
    if not accepted:
        verify.side_effect = ValueError("bad quality")
    path = tmp_path / "state.json"
    acceptance.run_scenario("retrieval", path, state, operation, verify)
    acceptance.run_scenario("retrieval", path, state, operation, verify)
    operation.assert_called_once()
    assert state["scenarios"]["retrieval"]["status"] == ("accepted" if accepted else "rejected")
    assert json.loads(path.read_text())["scenarios"]["retrieval"]["result"]["answer"] == "fixed"


def test_crashed_or_failed_scenario_cannot_silently_start_again(acceptance, tmp_path):
    state = {"scenarios": {"retrieval": {"status": "running"}}}
    operation = Mock()
    acceptance.run_scenario("retrieval", tmp_path / "state.json", state, operation, Mock())
    operation.assert_not_called()
    state = {"scenarios": {}}
    operation.side_effect = RuntimeError("private capability must not be serialized")
    path = tmp_path / "state.json"
    acceptance.run_scenario("retrieval", path, state, operation, Mock())
    acceptance.run_scenario("retrieval", path, state, operation, Mock())
    operation.assert_called_once()
    assert state["scenarios"]["retrieval"]["status"] == "failed"
    assert "capability" not in path.read_text()


def test_saved_configuration_rejects_changed_settings(acceptance, tmp_path):
    settings = {"runner": {"image": "hermes:first"}}
    path = tmp_path / "state.json"
    initial = acceptance.load_state(path, settings)
    acceptance.save_json(path, initial)
    assert acceptance.load_state(path, settings) == initial
    assert initial["settings"] == settings
    with pytest.raises(ValueError, match="configuration"):
        acceptance.load_state(path, {"runner": {"image": "hermes:changed"}})


def test_legacy_state_discards_checksums_and_preserves_answer(acceptance, tmp_path):
    settings = {"runner": {"image": "hermes:first"}}
    initial = {
        "settings": {**settings, "metric_lock_sha256": "unusable"},
        "configuration_sha256": "unusable", "fixture_sha256": "unusable",
        "scenarios": {"retrieval": {"status": "accepted", "result": {"answer": "saved"}}},
    }
    path = tmp_path / "state.json"
    acceptance.save_json(path, initial)
    loaded = acceptance.load_state(path, settings)
    assert loaded == {"settings": settings, "scenarios": initial["scenarios"]}
    operation = Mock()
    acceptance.run_scenario("retrieval", path, loaded, operation, Mock())
    operation.assert_not_called()
    with pytest.raises(ValueError, match="configuration"):
        acceptance.load_state(path, {"runner": {"image": "hermes:changed"}})


def test_acceptance_verification_does_not_require_metric_lock(acceptance, tmp_path, monkeypatch):
    from evaluator import metric_package

    settings = {"metric_package": str(tmp_path / "metric")}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "settings": settings,
        "scenarios": {"retrieval": {"status": "accepted", "result": {}}},
    }))
    load_metric = Mock()
    verify_result = Mock()
    monkeypatch.setattr(metric_package, "load_metric", load_metric)
    monkeypatch.setitem(acceptance.main.__globals__, "verify_result", verify_result)
    monkeypatch.setattr(acceptance.sys, "argv", [
        "run.py", "--config", str(config), "--state", str(state),
        "--scenario", "retrieval", "--verify-only",
    ])
    assert acceptance.main() == 0
    load_metric.assert_called_once_with(tmp_path / "metric")
    verify_result.assert_called_once_with(settings, "retrieval", {})
