"""Planning and full retrieval contracts; all model replies are synthetic."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def api():
    sys.path.insert(0, str(PACKAGE))
    spec = importlib.util.spec_from_file_location("new_planned_candidate", PACKAGE / "candidate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_corpus(tmp_path):
    kb = tmp_path / "norm"
    kb.mkdir()
    docs = {
        "a" * 64: "# Порядок\n1. Заявку на оборудование оформляет работник.\n",
        "b" * 64: "# Регламент\n1. Согласование приобретения осуществляет комиссия.\n",
        "c" * 64: "# Сроки\n1. Публикация сведений производится по понедельникам.\n",
        "d" * 64: "# Погода\n1. Снег и дождь.\n",
    }
    for origin, text in docs.items():
        (kb / f"{origin}.md").write_bytes(text.encode())
    return kb, docs


def test_plan_cannot_invent_numbers_and_keeps_original_identifiers(api):
    question = "Порядок закупки по приказу № 123-1 от 01.09.2026?"
    queries = api.validate_plan({"subqueries": ["Согласование приобретения"]}, question, 3, 600)
    assert "123-1" in queries[0] and "01.09.2026" in queries[0]
    with pytest.raises(ValueError, match="identifier"):
        api.validate_plan({"subqueries": ["Приказ № 777-1"]}, question, 3, 600)


@pytest.mark.parametrize(
    "bad",
    [
        {"answer": "Комиссия"},
        {"subqueries": "строка"},
        {"subqueries": [True]},
        {"subqueries": ["a", "b", "c", "d"]},
        {"subqueries": ["   "]},
    ],
)
def test_bad_plans_fail_explicitly(api, bad):
    with pytest.raises(ValueError):
        api.validate_plan(bad, "Кто согласует закупку?", 3, 600)


def test_duplicate_queries_do_not_receive_extra_rrf_votes(api):
    queries = api.validate_plan(
        {"subqueries": ["Согласование закупки", " согласование   закупки "]},
        "Кто одобряет покупку?",
        3,
        600,
    )
    assert len(queries) == 1


def test_live_planner_gets_question_only_once_and_not_gold(api, monkeypatch):
    calls = []

    def fake(url, payload, key_env, timeout):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": '{"subqueries":["Согласование закупки"]}'}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 7},
        }

    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    monkeypatch.setattr(api.base, "post_json", fake)
    plan = api.create_plan("Кто одобряет покупку?", api.defaults()["planner"])
    assert len(calls) == 1 and plan["model_calls"] == 1
    assert plan["queries"] == ["Согласование закупки"]
    assert json.loads(calls[0]["messages"][1]["content"]) == {"question": "Кто одобряет покупку?"}
    assert plan["model"] == "synthetic" and plan["usage"]["completion_tokens"] == 7
    assert plan["question"] == "Кто одобряет покупку?"
    assert plan["prompt"] == api.PLANNER_PROMPT


def test_malformed_planner_reply_is_not_silently_replaced_or_retried(api, monkeypatch):
    calls = []

    def fake(*args):
        calls.append(1)
        return {"choices": [{"message": {"content": "Не JSON"}}]}

    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    monkeypatch.setattr(api.base, "post_json", fake)
    with pytest.raises(ValueError, match="plan"):
        api.create_plan("Кто согласует закупку?", api.defaults()["planner"])
    assert calls == [1]


def test_plan_adds_sources_and_original_question_always_runs(api, tmp_path):
    kb, docs = make_corpus(tmp_path)
    output = tmp_path / "index"
    api.build(kb, output, api.defaults())
    question = "Кто оформляет заявку на оборудование?"
    plan = api.cached_plan(question, ["Согласование приобретения", "Публикация сведений"])
    result = api.retrieve(output, question, plan=plan)
    assert result["queries"][0] == question
    assert len(result["queries"]) == 3
    origins = {h["origin_id"] for h in result["hits"]}
    assert {"a" * 64, "b" * 64, "c" * 64} <= origins
    for h in result["hits"]:
        assert h["quote"] == docs[h["origin_id"]][h["char_start"] : h["char_end"]]
    assert result["context_chars_used"] <= 16000


def test_no_plan_ablation_needs_no_model_and_matches_original_bm25(api, tmp_path, monkeypatch):
    kb, _ = make_corpus(tmp_path)
    output = tmp_path / "index"
    api.build(kb, output, api.defaults())
    monkeypatch.delenv("SC71_CHAT_URL", raising=False)
    question = "Кто оформляет заявку на оборудование?"
    result = api.retrieve(output, question, no_plan=True)
    baseline = api.base.retrieve(output, question)
    assert [h["chunk_id"] for h in result["hits"]] == [h["chunk_id"] for h in baseline["hits"]]
    assert result["plan"]["model_calls"] == 0 and result["planning_enabled"] is False


def test_cached_plan_is_bound_to_question(api, tmp_path):
    kb, _ = make_corpus(tmp_path)
    output = tmp_path / "index"
    api.build(kb, output, api.defaults())
    plan = api.cached_plan("Один вопрос?", ["Согласование приобретения"])
    with pytest.raises(ValueError, match="question"):
        api.retrieve(output, "Другой вопрос?", plan=plan)


def test_cached_plan_is_bound_to_prompt(api, tmp_path):
    kb, _ = make_corpus(tmp_path)
    output = tmp_path / "index"
    api.build(kb, output, api.defaults())
    question = "Кто оформляет заявку на оборудование?"
    plan = api.cached_plan(question, ["Согласование приобретения"])
    plan["prompt"] = "Другой промпт"
    with pytest.raises(ValueError, match="prompt"):
        api.retrieve(output, question, plan=plan)


def test_synthetic_plan_fixture_matches_question_and_prompt(api, tmp_path):
    output = tmp_path / "index"
    api.build(PACKAGE / "examples" / "norm", output, api.defaults())
    question = (PACKAGE / "examples" / "question.txt").read_text().strip()
    plan = json.loads((PACKAGE / "examples" / "synthetic-plan.json").read_text())
    assert plan["question"] == question
    assert plan["prompt"] == api.PLANNER_PROMPT
    result = api.retrieve(output, question, plan=plan)
    assert result["plan"]["source"] == "cached"
    assert result["queries"] == [question, *plan["queries"]]


def test_answer_prompt_is_shared_and_plan_never_becomes_evidence(api, monkeypatch):
    captured = []

    def fake(url, payload, key_env, timeout):
        captured.append(payload)
        return {"choices": [{"message": {"content": "ОТВЕТ ОТСУТСТВУЕТ В БАЗЕ"}}]}

    monkeypatch.setattr(api.base, "post_json", fake)
    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    result = {
        "hits": [],
        "links": [],
        "plan": {"queries": ["UNVERIFIED PLANNER TEXT"]},
        "gold": "GOLD MUST NOT BE SENT",
    }
    api.base.answer_question("Вопрос?", result, "llm", 1)
    assert "UNVERIFIED PLANNER TEXT" not in json.dumps(captured)
    assert "GOLD MUST NOT BE SENT" not in json.dumps(captured)


def test_empty_provider_choices_preserve_failure_accounting(api, monkeypatch):
    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    monkeypatch.setattr(
        api.base, "post_json", lambda *args: {"choices": [], "usage": {"total_tokens": 12}}
    )
    with pytest.raises(api.PlanningError) as failure:
        api.create_plan("Кто согласует закупку?", api.defaults()["planner"])
    assert failure.value.diagnostics["model_calls"] == 1
    assert failure.value.diagnostics["usage"]["total_tokens"] == 12


def test_query_and_context_budgets_are_independent_of_plan_size(api, tmp_path):
    kb, _ = make_corpus(tmp_path)
    value = api.defaults()
    value["lexical"]["top_k"] = 2
    value["lexical"]["context_chars"] = 300
    output = tmp_path / "index"
    api.build(kb, output, value)
    question = "Кто оформляет заявку на оборудование?"
    plan = api.cached_plan(question, ["Согласование приобретения", "Публикация сведений"])
    result = api.retrieve(output, question, plan=plan)
    assert len(result["hits"]) <= 2
    assert result["context_chars_used"] <= 300
    assert len(result["queries"]) == 3
