import json
import runpy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from evaluator.contracts import Example
from evaluator.metric_package import freeze_metric, load_metric
from evaluator.metrics import MetricModelAccess, aggregate_results, evaluate_answer
from evaluator.reliability import empty_usage

PACKAGE = Path(__file__).parents[1] / "metric"
A = "a" * 64
B = "b" * 64
C = "c" * 64
QUOTE = "Заявление передают секретарю в течение пяти рабочих дней."


@pytest.fixture
def metric() -> Any:
    return SimpleNamespace(**runpy.run_path(str(PACKAGE / "score.py")))


@pytest.fixture
def kb(tmp_path: Path) -> Path:
    (tmp_path / "norm").mkdir()
    for origin in (A, B):
        (tmp_path / "norm" / f"{origin}.md").write_text(QUOTE, encoding="utf-8")
    return tmp_path


def payload(answer: str | None = None, *, topic: str = "T1", answerable: bool = True) -> dict:
    gold: dict[str, Any] = {
        "topic": topic,
        "answerable": answerable,
        "citations": [{
            "origin_id": A, "equivalent_origin_ids": [B], "role": "primary",
            "quote": QUOTE, "locator": "п. 1",
        }],
        "must_include": ["пяти рабочих дней"],
        "must_not_include": ["десяти рабочих дней"],
    }
    if topic == "T3":
        gold["t3"] = {"steps": [gold["citations"][0]]}
    if topic == "T4":
        gold["t4"] = {
            "collision_type": "amendment", "applicable": "both",
            "norm_a": gold["citations"][0],
            "norm_b": {"origin_id": B, "quote": QUOTE, "locator": "п. 2"},
        }
    if topic == "T5":
        gold["t5"] = {"target_origin_id": A}
    reference = {
        "format": "sc71-reference-v1", "topic": topic,
        "reference_markdown": "СЕКРЕТНЫЙ ЭТАЛОН", "gold": gold,
    }
    return {
        "example": {"id": "synthetic-1", "input": "СЕКРЕТНЫЙ ВОПРОС",
                    "reference": json.dumps(reference, ensure_ascii=False)},
        "answer": answer if answer is not None else f"«{QUOTE}» [/kb/norm/{A}.md, п. 1]",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "model_base_url": "http://gateway/runs/test/capability/v1",
        "model_request_timeout": 130410,
    }


def vote(score: int) -> str:
    return json.dumps({"score": score, "reason": "СЕКРЕТНОЕ ОБОСНОВАНИЕ"})


def test_weighted_nine_independent_votes_and_no_text_in_statistics(metric: Any, kb: Path) -> None:
    complete = Mock(side_effect=[vote(x) for x in (5, 4, 3, 4, 4, 4, 2, 3, 4)])
    result = metric.evaluate(payload(), kb, complete)
    assert result["score"] == pytest.approx(3.7)
    assert complete.call_count == 9
    assert result["statistics"]["topic"] == "T1"
    assert result["statistics"]["citation_recall"] == 1
    assert result["statistics"]["fabricated_quote"] is False
    serialized = json.dumps(result, ensure_ascii=False)
    assert "СЕКРЕТ" not in serialized
    assert QUOTE not in serialized
    assert A not in serialized
    assert [call.args[0]["trace_metadata"] for call in complete.call_args_list] == [
        {"role": "judge", "criterion": criterion, "vote": index}
        for criterion in ("completeness", "accuracy", "groundedness")
        for index in (1, 2, 3)
    ]
    for call in complete.call_args_list:
        request = call.args[0]
        assert request["model"] == payload()["model"]
        assert request["reasoning_effort"] == "max"
        assert request["response_schema"]["properties"]["score"] == {
            "type": "integer", "minimum": 1, "maximum": 5,
        }
        assert request["response_schema"]["properties"]["reason"]["pattern"] == r"\S"
        assert request["response_schema"]["additionalProperties"] is False
        assert request["response_schema"] == metric.SCORE_SCHEMA
        assert len(request["messages"]) == 2
        data = json.loads(request["messages"][1]["content"])
        assert "weight" not in data["criterion"]
        assert "votes" not in data
        assert data["question"] == "СЕКРЕТНЫЙ ВОПРОС"


@pytest.mark.parametrize("answer,answerable,flag", [
    ("", True, "candidate_failure"),
    (f"Ответ [/kb/norm/{C}.md]", True, "nonexistent_citation"),
    ("Несуществующая норма есть", False, "hallucination"),
])
def test_only_defined_zero_gates_skip_judge(
    metric: Any, kb: Path, answer: str, answerable: bool, flag: str,
) -> None:
    complete = Mock(side_effect=AssertionError("must not call model"))
    result = metric.evaluate(payload(answer, answerable=answerable), kb, complete)
    assert result["score"] == 0
    assert result["statistics"][flag] is True
    complete.assert_not_called()


def test_normalized_refusal_is_judged_and_false_refusal_is_not_zero_gate(
    metric: Any, kb: Path,
) -> None:
    answer = "\n Ответ   отсутствует в базе.\nНедостаточно сведений."
    for answerable in (False, True):
        complete = Mock(return_value=vote(3))
        result = metric.evaluate(payload(answer, answerable=answerable), kb, complete)
        assert result["score"] == pytest.approx(3)
        assert result["statistics"]["abstained"] is True
        assert result["statistics"]["false_abstention"] is answerable
        assert complete.call_count == 9


def test_equivalent_ids_case_and_whitespace_match(metric: Any, kb: Path) -> None:
    answer = f'"{QUOTE.upper().replace(" ", "  ")}" [/kb/norm/{B}.md, п. 1]'
    result = metric.evaluate(payload(answer), kb, Mock(return_value=vote(5)))
    stats = result["statistics"]
    assert stats["citation_recall"] == 1
    assert stats["citation_precision"] == 1
    assert stats["fabricated_quote"] is False
    assert stats["must_include_hit"] == 1


def test_fabricated_quote_is_diagnostic_not_an_extra_gate(metric: Any, kb: Path) -> None:
    answer = f"«Совершенно выдуманная цитата с неверным сроком.» [/kb/norm/{A}.md]"
    complete = Mock(return_value=vote(2))
    result = metric.evaluate(payload(answer), kb, complete)
    assert result["score"] == 2
    assert result["statistics"]["fabricated_quote"] is True
    assert complete.call_count == 9


def test_missing_corpus_and_gold_documents_are_errors(metric: Any, kb: Path) -> None:
    with pytest.raises(ValueError, match="corpus"):
        metric.evaluate(payload(), kb / "missing", Mock())
    (kb / "norm" / f"{A}.md").unlink()
    with pytest.raises(ValueError, match="reference document"):
        metric.evaluate(payload("Some answer"), kb, Mock())


@pytest.mark.parametrize("response", [
    '{"score":true,"reason":"x"}', '{"score":0,"reason":"x"}',
    '{"score":6,"reason":"x"}', '{"score":3.5,"reason":"x"}',
    '{"score":3,"reason":""}', '{"score":3,"reason":"x","extra":1}', "bad JSON",
])
def test_invalid_judge_vote_is_error_without_hidden_retry(
    metric: Any, kb: Path, response: str,
) -> None:
    complete = Mock(return_value=response)
    with pytest.raises(ValueError):
        metric.evaluate(payload(), kb, complete)
    assert complete.call_count == 1


def test_judge_transport_error_is_not_zero_or_retried(metric: Any, kb: Path) -> None:
    complete = Mock(side_effect=TimeoutError("gateway timed out"))
    with pytest.raises(TimeoutError):
        metric.evaluate(payload(), kb, complete)
    assert complete.call_count == 1


def test_topic_specific_diagnostics(metric: Any, kb: Path) -> None:
    complete = Mock(return_value=vote(4))
    t1 = metric.evaluate(payload(f"десяти рабочих дней [/kb/norm/{A}.md]"), kb, complete)
    assert t1["statistics"]["t1_update_error"] is True
    t3 = metric.evaluate(payload(f"[/kb/norm/{B}.md]", topic="T3"), kb, complete)
    assert t3["statistics"]["citation_recall"] == 1
    answer = json.dumps({"collision_type": "amendment", "applicable_norm": f"/kb/norm/{B}.md"})
    t4 = metric.evaluate(payload(answer, topic="T4"), kb, complete)
    assert t4["statistics"]["collision_type_match"] is True
    assert t4["statistics"]["applicable_norm_match"] is True
    t5 = metric.evaluate(payload(f"[/kb/norm/{B}.md]", topic="T5"), kb, complete)
    assert t5["statistics"]["target_hit"] is True


def test_topic_macro_average_handles_unequal_counts_and_subsets(metric: Any) -> None:
    results = [
        {"score": 5, "statistics": {"topic": "T1"}},
        {"score": 3, "statistics": {"topic": "T1"}},
        {"score": 0, "statistics": {"topic": "T2"}},
    ]
    assert metric.aggregate(results) == {"score": 2}


@pytest.mark.parametrize("results", [
    [], [{"score": True, "statistics": {"topic": "T1"}}],
    [{"score": 6, "statistics": {"topic": "T1"}}],
    [{"score": float("nan"), "statistics": {"topic": "T1"}}],
    [{"score": 3, "statistics": {"topic": "T0"}}],
    [{"score": 3}],
])
def test_aggregation_rejects_missing_or_invalid_measurements(metric: Any, results: list) -> None:
    with pytest.raises((ValueError, KeyError)):
        metric.aggregate(results)


def test_wrong_model_and_reference_shape_fail_before_request(metric: Any, kb: Path) -> None:
    item = payload()
    item["model"] = "other-model"
    with pytest.raises(ValueError, match="model"):
        metric.evaluate(item, kb, Mock())
    item = payload()
    item["example"]["reference"] = "plain text"
    with pytest.raises(ValueError):
        metric.evaluate(item, kb, Mock())


@pytest.mark.parametrize("answer", [
    "Пояснение\nОТВЕТ ОТСУТСТВУЕТ В БАЗЕ",
    "Вероятно, ОТВЕТ ОТСУТСТВУЕТ В БАЗЕ",
])
def test_refusal_must_be_the_first_entire_nonempty_line(
    metric: Any, kb: Path, answer: str,
) -> None:
    complete = Mock(side_effect=AssertionError("must not call model"))
    result = metric.evaluate(payload(answer, answerable=False), kb, complete)
    assert result["score"] == 0
    assert result["statistics"]["abstained"] is False


def test_quotes_in_additional_cited_documents_are_valid(metric: Any, kb: Path) -> None:
    extra_quote = "Другая норма устанавливает обязанность приложить копию заявления."
    (kb / "norm" / f"{C}.md").write_text(extra_quote, encoding="utf-8")
    answer = f"«{extra_quote}» [/kb/norm/{C}.md]"
    result = metric.evaluate(payload(answer), kb, Mock(return_value=vote(4)))
    assert result["statistics"]["fabricated_quote"] is False
    assert result["statistics"]["citation_precision"] == 0
    uncited = metric.evaluate(payload(f"«{extra_quote}» [/kb/norm/{A}.md]"), kb,
                              Mock(return_value=vote(4)))
    assert uncited["statistics"]["fabricated_quote"] is True


def test_json_keys_and_nonquote_strings_are_not_fabricated_quotes(metric: Any, kb: Path) -> None:
    answer = json.dumps({
        "collision_type": "amendment", "applicable_norm": {"origin_id": B},
        "explanation": "Это собственное пояснение ответа, а не дословная цитата.",
        "source": {"quote": QUOTE, "citation": f"/kb/norm/{B}.md"},
    }, ensure_ascii=False)
    result = metric.evaluate(payload(f"```json\n{answer}\n```", topic="T4"), kb,
                             Mock(return_value=vote(4)))
    assert result["statistics"]["fabricated_quote"] is False
    assert result["statistics"]["quote_count"] == 1
    assert result["statistics"]["applicable_norm_match"] is True


def test_duplicate_references_and_case_are_normalized(metric: Any, kb: Path) -> None:
    answer = f"[/kb/norm/{A.upper()}.md] [/kb/norm/{A}.md]"
    result = metric.evaluate(payload(answer), kb, Mock(return_value=vote(4)))
    assert result["statistics"]["cited_document_count"] == 1
    assert result["statistics"]["citation_recall"] == 1


def test_late_judge_failure_does_not_average_partial_votes(metric: Any, kb: Path) -> None:
    complete = Mock(side_effect=[vote(5), vote(5), vote(5), vote(5), "invalid"])
    with pytest.raises(ValueError):
        metric.evaluate(payload(), kb, complete)
    assert complete.call_count == 5


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_http_uses_gateway_timeout_schema_and_no_proxy_or_sdk_retries(
    metric: Any, monkeypatch: pytest.MonkeyPatch, finish_reason: str | None,
) -> None:
    choice: dict[str, Any] = {"message": {"content": vote(4)}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    response = BytesIO(json.dumps({"choices": [choice]}).encode())
    opener = Mock()
    opener.open.return_value = response
    build = Mock(return_value=opener)
    monkeypatch.setattr(metric.urllib.request, "build_opener", build)
    body = {"model": payload()["model"], "response_schema": {"type": "object"}}
    assert metric.complete_http(payload(), body) == vote(4)
    assert opener.open.call_count == 1
    assert build.call_args.args[0].proxies == {}
    request = opener.open.call_args.args[0]
    assert request.full_url == payload()["model_base_url"] + "/chat/completions"
    assert json.loads(request.data) == body
    assert opener.open.call_args.kwargs["timeout"] == 130410


def test_frozen_package_runs_in_evaluator_with_zero_model_usage(
    kb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = tmp_path / "metric"
    freeze_metric(PACKAGE, frozen)
    metric = load_metric(frozen)
    monkeypatch.setenv("ARENA_SC71_KB_ROOT", str(kb))
    item = payload("", topic="T2")
    usage = empty_usage()
    result = evaluate_answer(
        metric, Example.model_validate(item["example"]), item["answer"],
        model_access=MetricModelAccess(
            model=item["model"], base_url="http://127.0.0.1:9/v1",
            usage_dir=tmp_path / "usage",
        ), usage=usage,
    )
    assert result["score"] == 0
    assert usage == empty_usage()
    assert aggregate_results(metric, [result, {"score": 4, "statistics": {"topic": "T1"}}]) == 2
