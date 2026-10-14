import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from evaluator.contracts import Criterion, Example, Metric
from evaluator.judge import ModelClient, judge_answer


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)


def response_payload(content: str, **extra: Any) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        **extra,
    }


@pytest.fixture
def metric() -> Metric:
    return Metric(
        version="v1",
        instruction="Оцени качество результата.",
        criteria=[
            Criterion(id="correct", question="Верны ли выводы?", weight=3),
            Criterion(id="complete", question="Все ли требования выполнены?", weight=1),
        ],
    )


@pytest.fixture
def example() -> Example:
    return Example(id="sample", input="SECRET_SOURCE", reference="Замороженный эталон")


def test_three_independent_votes_per_criterion(metric: Metric, example: Example) -> None:
    requests: list[list[dict[str, str]]] = []
    scores = iter([1, 3, 5, 2, 2, 2])

    def complete(messages: list[dict[str, str]]) -> str:
        requests.append(messages)
        return json.dumps({"score": next(scores), "reason": "Проверенное обоснование"})

    result = judge_answer(metric, example, "Ответ кандидата", complete)

    assert result["score"] == pytest.approx(2.75)
    assert result["criteria"] == {
        "correct": {
            "scores": [1, 3, 5],
            "reasons": ["Проверенное обоснование"] * 3,
        },
        "complete": {
            "scores": [2, 2, 2],
            "reasons": ["Проверенное обоснование"] * 3,
        },
    }
    assert len(requests) == 6
    assert len({id(messages) for messages in requests}) == 6
    assert requests[0] == requests[1] == requests[2]
    assert requests[3] == requests[4] == requests[5]
    for index, messages in enumerate(requests):
        assert [message["role"] for message in messages] == ["system", "user"]
        assert metric.instruction in messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert payload == {
            "criterion": {
                "id": metric.criteria[index // 3].id,
                "question": metric.criteria[index // 3].question,
            },
            "reference": example.reference,
            "answer": "Ответ кандидата",
        }
        assert "SECRET_SOURCE" not in json.dumps(messages)
        assert "Проверенное обоснование" not in json.dumps(messages, ensure_ascii=False)


@pytest.mark.parametrize(
    "vote",
    [
        '{"score": true, "reason": "ok"}',
        '{"score": 3.0, "reason": "ok"}',
        '{"score": "3", "reason": "ok"}',
        '{"score": 0, "reason": "ok"}',
        '{"score": 6, "reason": "ok"}',
        '{"score": 3, "reason": ""}',
        '{"score": 3, "reason": "   "}',
        '{"score": 3, "reason": "ok", "extra": 1}',
        '```json\n{"score": 3, "reason": "ok"}\n```',
        'Result: {"score": 3, "reason": "ok"}',
        '{"score": 3}',
    ],
)
def test_invalid_vote_fails_without_inventing_score(
    metric: Metric, example: Example, vote: str
) -> None:
    with pytest.raises(ValueError):
        judge_answer(metric, example, "Ответ", lambda _: vote)


@pytest.mark.parametrize("answer", ["", " \n\t"])
def test_empty_answer_is_not_sent_to_judge(metric: Metric, example: Example, answer: str) -> None:
    def complete(_: list[dict[str, str]]) -> str:
        pytest.fail("Empty answer must not be judged")

    with pytest.raises(ValueError, match="answer"):
        judge_answer(metric, example, answer, complete)


def test_weights_are_normalized_without_overflow(example: Example) -> None:
    metric = Metric(
        version="v1",
        instruction="Оцени.",
        criteria=[
            Criterion(id="a", question="A?", weight=1e308),
            Criterion(id="b", question="B?", weight=1e308),
        ],
    )
    result = judge_answer(metric, example, "Ответ", lambda _: '{"score": 4, "reason": "ok"}')
    assert result["score"] == 4


def install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[dict[str, Any]]:
    client_class = httpx.Client
    options: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> httpx.Client:
        options.append(kwargs)
        return client_class(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("evaluator.judge.httpx.Client", client)
    return options


def test_client_uses_exact_endpoint_and_fixed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json=response_payload("Ответ")
        )

    options = install_transport(monkeypatch, handler)
    messages = [{"role": "user", "content": "Задача"}]
    client = ModelClient("http://internal-model/v1/", "fixed", "secret", {"temperature": 0.2}, 7)

    assert client.complete(messages) == "Ответ"
    assert len(requests) == 1
    assert str(requests[0].url) == "http://internal-model/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content) == {
        "model": "fixed",
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False
    assert options[0]["timeout"] == 7


@pytest.mark.parametrize("key", ["model", "messages", "stream"])
def test_parameters_cannot_replace_request_contract(key: str) -> None:
    with pytest.raises(ValueError):
        ModelClient("http://internal-model/v1", "fixed", parameters={key: "override"})


def test_client_does_not_follow_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://external.invalid/v1"})

    install_transport(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        ModelClient("http://internal-model/v1", "fixed").complete([])
    assert requests == ["http://internal-model/v1/chat/completions"] * 6


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": " "}}]},
        {"choices": [{"message": {"content": [{"text": "text"}]}}]},
        {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "cut"}, "finish_reason": "content_filter"}]},
    ],
)
def test_client_rejects_invalid_completion(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(ValueError):
        ModelClient("http://internal-model/v1", "fixed").complete([])


def test_client_accepts_completion_without_optional_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json=response_payload("ok", choices=[{"message": {"content": "ok"}}]),
        ),
    )
    assert ModelClient("http://internal-model/v1", "fixed").complete([]) == "ok"


def test_client_rejects_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(monkeypatch, lambda _: httpx.Response(200, content=b"x" * (2**21 + 1)))
    with pytest.raises(ValueError, match="size"):
        ModelClient("http://internal-model/v1", "fixed").complete([])


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://internal-model/v1",
        "http://user:secret@internal-model/v1",
        "http://internal-model/v1?target=other",
        "http://internal-model/v1#part",
        "not-a-url",
    ],
)
def test_client_rejects_ambiguous_endpoints(base_url: str) -> None:
    with pytest.raises(ValueError):
        ModelClient(base_url, "fixed")


def test_client_retries_transport_then_malformed_completion_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("slow", request=request)
        content = "" if attempts == 2 else "recovered"
        return httpx.Response(200, json=response_payload(content))

    install_transport(monkeypatch, handler)
    client = ModelClient("http://internal-model/v1", "fixed")
    assert client.complete([]) == "recovered"
    assert attempts == 3
    assert client.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "attempts": 3,
        "unreported_attempts": 1,
    }


def test_client_has_exactly_five_retries_and_retains_unknown_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("slow", request=request)

    install_transport(monkeypatch, handler)
    client = ModelClient("http://internal-model/v1", "fixed")
    with pytest.raises(httpx.ReadTimeout):
        client.complete([])
    assert attempts == 6
    assert sleeps == [30, 60, 120, 240, 300]
    assert client.usage["attempts"] == client.usage["unreported_attempts"] == 6
    assert client.elapsed_seconds >= 0


@pytest.mark.parametrize("bound_method", [False, True])
def test_judge_validation_shares_one_retry_budget_with_client(
    monkeypatch: pytest.MonkeyPatch, metric: Metric, example: Example, bound_method: bool
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls % 2:
            return httpx.Response(503)
        return httpx.Response(200, json=response_payload('{"score": 8, "reason": "bad"}'))

    install_transport(monkeypatch, handler)
    client = ModelClient("http://internal-model/v1", "fixed")
    with pytest.raises(ValueError):
        judge_answer(metric, example, "Ответ", client.complete if bound_method else client)
    assert calls == 6
    assert client.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
        "attempts": 6,
        "unreported_attempts": 3,
    }


def test_judge_retries_invalid_vote_without_repeating_successful_votes(
    monkeypatch: pytest.MonkeyPatch, metric: Metric, example: Example
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "invalid" if calls == 2 else '{"score": 4, "reason": "ok"}'
        return httpx.Response(200, json=response_payload(content))

    install_transport(monkeypatch, handler)
    client = ModelClient("http://internal-model/v1", "fixed")
    result = judge_answer(metric, example, "Ответ", client)
    assert result["score"] == 4
    assert calls == 7
    assert client.usage["total_tokens"] == 105


def test_injected_judge_callable_retries_vote_validation(metric: Metric, example: Example) -> None:
    calls = 0

    def complete(_: list[dict[str, str]]) -> str:
        nonlocal calls
        calls += 1
        return "invalid" if calls == 1 else '{"score": 4, "reason": "ok"}'

    assert judge_answer(metric, example, "Ответ", complete)["score"] == 4
    assert calls == 7


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": 5}])
def test_client_requires_measured_token_usage(
    monkeypatch: pytest.MonkeyPatch, usage: Any
) -> None:
    install_transport(
        monkeypatch, lambda _: httpx.Response(200, json=response_payload("ok", usage=usage))
    )
    client = ModelClient("http://internal-model/v1", "fixed")
    with pytest.raises(ValueError, match="usage"):
        client.complete([])
    assert client.usage["attempts"] == client.usage["unreported_attempts"] == 6


def test_default_timeout_allows_slow_models(monkeypatch: pytest.MonkeyPatch) -> None:
    options = install_transport(
        monkeypatch, lambda _: httpx.Response(200, json=response_payload("ok"))
    )
    client = ModelClient("http://internal-model/v1", "fixed")
    client.complete([])
    assert options[0]["timeout"] == 21600


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_client_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError):
        ModelClient("http://internal-model/v1", "fixed", timeout=timeout)


def test_http_failure_still_counts_reported_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(
        monkeypatch, lambda _: httpx.Response(503, json=response_payload("unavailable"))
    )
    client = ModelClient("http://internal-model/v1", "fixed")
    with pytest.raises(httpx.HTTPStatusError):
        client.complete([])
    assert client.usage["attempts"] == 6
    assert client.usage["total_tokens"] == 90
    assert client.usage["unreported_attempts"] == 0


def test_custom_validator_is_retried_with_measured_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=response_payload("ok")))
    validations = 0

    def validator(_: str) -> None:
        nonlocal validations
        validations += 1
        if validations < 3:
            raise ValueError("bad format")

    client = ModelClient("http://internal-model/v1", "fixed")
    assert client.complete([], validator=validator) == "ok"
    assert client.usage["attempts"] == 3
    assert client.usage["total_tokens"] == 45
