import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from evaluator.contracts import Example
from evaluator.gateway import GatewaySettings, create_gateway


@pytest.fixture(autouse=True)
def retry_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr("evaluator.gateway.sleep", sleep, raising=False)
    return waits


def valid_response(answer: str = "Answer") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def request_body() -> dict[str, object]:
    return {"model": "fixed-model", "messages": [{"role": "user", "content": "Solve this"}]}


def test_gateway_forwards_fixed_configuration_and_sanitizes_response() -> None:
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "private-id",
                "choices": [
                    {"message": {"content": "Answer", "reasoning_content": "model reasoning"}}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                "debug": "private upstream state",
            },
        )

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model:8000/v1",
            model="fixed-model",
            api_key="operator-key",
            parameters={"temperature": 0.2, "max_tokens": 128},
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", json=request_body())

    assert result.status_code == 200
    assert result.json() == {
        "choices": [{"message": {"content": "Answer", "reasoning_content": "model reasoning"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }
    assert len(requests) == 1
    assert str(requests[0].url) == "http://internal-model:8000/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer operator-key"
    assert json.loads(requests[0].content) == {
        **request_body(),
        "temperature": 0.2,
        "max_tokens": 128,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "different-model"},
        {"temperature": 1},
        {"max_tokens": 999},
        {"top_p": 1},
        {"tools": []},
        {"stream": True},
        {"logit_bias": {"1": 100}},
        {"base_url": "http://external"},
        {"messages": [{"role": "tool", "content": "input"}]},
        {"messages": [{"role": "user", "content": [{"image_url": "http://external"}]}]},
        {"messages": [{"role": "user", "content": "input", "name": "hidden"}]},
        {"messages": []},
    ],
)
def test_gateway_rejects_client_configuration(changes: dict[str, object]) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid requests must not reach the model")

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={**request_body(), **changes})
    assert response.status_code in (400, 422)
    assert response.json() == {"detail": "Invalid model request"}


@pytest.mark.parametrize("status", [301, 302, 307, 308, 400, 401, 429, 500])
def test_gateway_does_not_follow_redirects_or_expose_errors(status: int) -> None:
    calls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            status,
            headers={"location": "https://external.example/private"},
            text="secret upstream details",
        )

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}
    assert calls == ["http://internal-model/v1/chat/completions"] * 6


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": {"secret": "state"}}}]},
    ],
)
def test_gateway_rejects_malformed_upstream_response(payload: dict[str, object]) -> None:
    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={**valid_response(), **payload})
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}


def test_gateway_limits_request_and_response_before_parsing() -> None:
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=valid_response("x" * 256))

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            max_body_bytes=256,
            max_response_bytes=256,
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        oversized = client.post("/v1/chat/completions", content=b"x" * 257)
        malformed = client.post("/v1/chat/completions", content=b"not JSON")
        too_large_response = client.post("/v1/chat/completions", json=request_body())
    assert oversized.status_code == 413
    assert malformed.status_code == 400
    assert too_large_response.status_code == 502
    assert len(calls) == 6


@pytest.mark.parametrize("character", ["\x00", "я", "😀"])
def test_gateway_accepts_maximum_input_prompt_and_context(tmp_path: Path, character: str) -> None:
    from evaluator.usage import create_usage, read_usage

    example = Example(id="maximum-input", input=character * 500000)
    prompt = "\x00" * (1024 * 1024)
    context = "\u0080" * ((1024 * 1024 - len('{"context":""}')) // 2)
    payload = {
        "model": "fixed-model",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Task context:\n" + context},
            {"role": "user", "content": example.input},
        ],
    }
    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert json.loads(request.content) == payload
        return httpx.Response(200, json=valid_response())

    settings = GatewaySettings(
        base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
    )
    app = create_gateway(settings, transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions",
            content=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 200
    assert response.json() == valid_response()
    assert calls == 1
    assert read_usage(tmp_path, run_id) == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "attempts": 1,
        "unreported_attempts": 0,
    }
    assert settings.max_body_bytes == 16 * 1024 * 1024
    assert settings.max_response_bytes == 2 * 1024 * 1024


def test_gateway_limits_request_and_response_independently() -> None:
    body = json.dumps(request_body()).encode()
    output = json.dumps(valid_response()).encode()
    calls = 0
    oversized_response = False

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=output + (b" " if oversized_response else b""))

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            max_body_bytes=len(body),
            max_response_bytes=len(output),
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=body)
        assert response.status_code == 200
        assert response.json() == valid_response()
        assert calls == 1
        assert client.post("/v1/chat/completions", content=body + b" ").status_code == 413
        assert calls == 1
        oversized_response = True
        assert client.post("/v1/chat/completions", content=body).status_code == 502
    assert calls == 7


@pytest.mark.parametrize(
    ("max_body_bytes", "max_response_bytes"), [(0, 1), (-1, 1), (1, 0), (1, -1)]
)
def test_gateway_rejects_invalid_size_limits(max_body_bytes: int, max_response_bytes: int) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            max_body_bytes=max_body_bytes,
            max_response_bytes=max_response_bytes,
        )


def test_gateway_exposes_no_state_or_api_documentation() -> None:
    app = create_gateway(GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        for path in ("/v1/models", "/history", "/logs", "/openapi.json", "/docs", "/redoc"):
            assert client.get(path).status_code == 404


def test_gateway_loads_operator_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "model-key"
    key.write_text("private-token\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_BASE_URL", "http://internal-model:8000/v1/")
    monkeypatch.setenv("MODEL_NAME", "fixed-model")
    monkeypatch.setenv("MODEL_API_KEY_FILE", str(key))
    monkeypatch.setenv("MODEL_PARAMETERS_JSON", '{"temperature":0,"top_p":1,"max_tokens":512}')
    settings = GatewaySettings.from_env()
    assert settings.base_url == "http://internal-model:8000/v1"
    assert settings.model == "fixed-model"
    assert settings.api_key == "private-token"
    assert settings.parameters == {"temperature": 0, "top_p": 1, "max_tokens": 512}
    assert "private-token" not in repr(settings)


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///etc/passwd",
        "ftp://model/v1",
        "http://user:password@model/v1",
        "http://model/v1?path=other",
        "http://model/v1#fragment",
        "http://model",
    ],
)
def test_gateway_rejects_ambiguous_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(base_url=base_url, model="fixed-model")


@pytest.mark.parametrize("parameters", [{"model": "other"}, {"messages": []}, {"stream": True}])
def test_gateway_does_not_override_fixed_model_with_operator_parameters(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", parameters=parameters
        )


def test_gateway_bounds_entire_upstream_call_time() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", timeout_seconds=0.01
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}


@pytest.mark.parametrize("finish_reason", [{}, {"finish_reason": "stop"}])
def test_gateway_accepts_response_without_usage(finish_reason: dict[str, str]) -> None:
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "complete"}, **finish_reason}]},
        )

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "complete"
    assert response.json()["usage"] is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "finish_reason", ["length", "content_filter", "tool_calls", "unknown", None]
)
def test_gateway_rejects_incomplete_responses(finish_reason: str | None) -> None:
    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    **valid_response(),
                    "choices": [
                        {"message": {"content": "partial"}, "finish_reason": finish_reason}
                    ],
                },
            )
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}


def test_gateway_forwards_operator_specific_model_parameters() -> None:
    parameters: dict[str, object] = {
        "reasoning_effort": "high",
        "max_completion_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=valid_response("complete"))

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", parameters=parameters
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    assert json.loads(requests[0].content) == {**request_body(), **parameters}


@pytest.mark.parametrize(
    "parameters",
    [{"seed": float("nan")}, {"extra": {"nested": float("inf")}}, {"extra": b"not JSON"}],
)
def test_gateway_rejects_non_json_operator_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", parameters=parameters
        )


@pytest.mark.parametrize("parameters", ["[]", "null", "1"])
def test_gateway_rejects_non_object_operator_parameters(
    monkeypatch: pytest.MonkeyPatch, parameters: str
) -> None:
    monkeypatch.setenv("MODEL_BASE_URL", "http://internal-model/v1")
    monkeypatch.setenv("MODEL_NAME", "fixed-model")
    monkeypatch.setenv("MODEL_PARAMETERS_JSON", parameters)
    with pytest.raises(ValueError):
        GatewaySettings.from_env()


def test_gateway_hides_transport_errors() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private upstream endpoint and credentials")

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}


def test_gateway_retries_transient_failures_and_preserves_request(retry_waits: list[float]) -> None:
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("slow model")
        if len(requests) == 2:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        if len(requests) == 3:
            return httpx.Response(200, json={**valid_response(), "choices": []})
        if len(requests) == 4:
            return httpx.Response(200, json=valid_response("   "))
        if len(requests) == 5:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    assert response.json() == {"choices": [{"message": {"content": "ok"}}], "usage": None}
    assert len(requests) == 5
    assert all(request.content == requests[0].content for request in requests)
    assert retry_waits == [30, 60, 120, 240]


def test_gateway_stops_after_five_retries(retry_waits: list[float]) -> None:
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=valid_response(""))

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert len(requests) == 6
    assert len(retry_waits) == 5


def test_gateway_reads_long_timeout_and_retry_delay_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODEL_BASE_URL", "http://internal-model/v1")
    monkeypatch.setenv("MODEL_NAME", "fixed-model")
    assert GatewaySettings.from_env().timeout_seconds == 21600
    monkeypatch.setenv("MODEL_TIMEOUT", "43200")
    monkeypatch.setenv("MODEL_RETRY_DELAY", "60")
    monkeypatch.setenv("MODEL_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TRACE_DIR", str(tmp_path / "traces"))
    settings = GatewaySettings.from_env()
    assert settings.timeout_seconds == 43200
    assert settings.retry_delay == 60
    assert settings.usage_dir == tmp_path
    assert settings.trace_dir == tmp_path / "traces"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_gateway_rejects_invalid_timeout(value: float) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", timeout_seconds=value
        )


def test_scoped_gateway_accounts_all_attempts_and_calls(tmp_path: Path) -> None:
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={**valid_response(), "choices": []})
        if calls == 2:
            raise httpx.ConnectError("internal issue")
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body()
            )
            assert response.status_code == 200
            assert response.json() == valid_response()
    assert read_usage(tmp_path, run_id) == {
        "prompt_tokens": 36,
        "completion_tokens": 9,
        "total_tokens": 45,
        "attempts": 4,
        "unreported_attempts": 1,
    }


def test_scoped_gateway_rejects_missing_scope_and_does_not_create_ledgers(tmp_path: Path) -> None:
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)

    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Missing scope must not reach model")

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=request_body()).status_code == 403
        for invalid_id in (str(uuid4()), "invalid-id", "%2e%2e"):
            assert (
                client.post(
                    f"/runs/{invalid_id}/{capability}/v1/chat/completions", json=request_body()
                ).status_code
                == 404
            )
        assert client.get(f"/runs/{run_id}/usage").status_code == 404
        assert (
            client.post(
                f"/runs/{run_id}/{capability}/v1/chat/completions",
                json={**request_body(), "model": "other"},
            ).status_code
            == 400
        )
    assert read_usage(tmp_path, run_id)["attempts"] == 0


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 14},
        {"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4},
    ],
)
def test_scoped_gateway_counts_unreported_usage_as_unknown(tmp_path: Path, usage: object) -> None:
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={**valid_response(), "usage": usage} if calls == 1 else valid_response()
        )

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body()
        )
    assert response.status_code == 200
    assert response.json()["usage"] is None
    assert calls == 1
    assert read_usage(tmp_path, run_id) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "attempts": 1,
        "unreported_attempts": 1,
    }


def test_scoped_gateway_rejects_sealed_scope(tmp_path: Path) -> None:
    from evaluator.usage import create_usage, seal_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    seal_usage(tmp_path, run_id)

    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Sealed scope must not reach model")

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body()
        )
    assert response.status_code == 404


@pytest.mark.parametrize("status", [200, 502])
def test_scoped_gateway_tracks_active_calls_until_all_attempts_end(
    tmp_path: Path, status: int, retry_waits: list[float]
) -> None:
    from evaluator.usage import create_usage, seal_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        with pytest.raises(ValueError):
            seal_usage(tmp_path, run_id)
        return httpx.Response(status, json=valid_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body()
        )
    assert response.status_code == status
    usage = seal_usage(tmp_path, run_id)
    assert usage["attempts"] == calls == 1
    assert usage["total_tokens"] == 15 * calls
    assert retry_waits == []


def test_scoped_gateway_stops_retrying_when_closed_during_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluator.usage import create_usage, seal_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = 0
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)
        with pytest.raises(ValueError):
            seal_usage(tmp_path, run_id)

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json=valid_response())

    monkeypatch.setattr("evaluator.gateway.sleep", sleep)
    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body()
        )
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}
    assert calls == 1
    assert waits == [30]
    usage = seal_usage(tmp_path, run_id)
    assert usage["attempts"] == 1
    assert usage["total_tokens"] == 15


def test_scoped_gateway_rejects_peer_uuid_without_changing_accounting(tmp_path: Path) -> None:
    from evaluator.usage import create_usage, read_usage, seal_usage

    victim_id = str(uuid4())
    peer_id = str(uuid4())
    victim_token = create_usage(tmp_path, victim_id)
    peer_token = create_usage(tmp_path, peer_id)
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    victim_path = tmp_path / f"{victim_id}.json"
    before = victim_path.read_bytes()
    with TestClient(app) as client:
        for endpoint in (
            f"/runs/{victim_id}/v1/chat/completions",
            f"/runs/{victim_id}/{peer_token}/v1/chat/completions",
            f"/runs/{victim_id}/{'0' * 64}/v1/chat/completions",
        ):
            response = client.post(endpoint, json=request_body())
            assert response.status_code == 404
            assert victim_path.read_bytes() == before
            assert calls == []
        response = client.post(
            f"/runs/{peer_id}/{peer_token}/v1/chat/completions", json=request_body()
        )
        assert response.status_code == 200
        assert len(calls) == 1
        assert read_usage(tmp_path, peer_id)["total_tokens"] == 15
        assert read_usage(tmp_path, victim_id)["total_tokens"] == 0
        assert seal_usage(tmp_path, victim_id)["attempts"] == 0
        response = client.post(
            f"/runs/{victim_id}/{victim_token}/v1/chat/completions", json=request_body()
        )
        assert response.status_code == 404
        assert len(calls) == 1


def test_gateway_entrypoint_does_not_log_capability_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluator import gateway

    options = {}
    monkeypatch.setattr(
        gateway.GatewaySettings,
        "from_env",
        lambda: GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
    )
    monkeypatch.setattr(gateway.uvicorn, "run", lambda app, **kwargs: options.update(kwargs))

    gateway.main()

    assert options["access_log"] is False


def tool_call() -> dict[str, object]:
    return {
        "id": "call-search",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query":"example"}'},
    }


def tool_request() -> dict[str, object]:
    return {
        **request_body(),
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the permitted documents",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        "tool_choice": "required",
    }


def tool_response() -> dict[str, Any]:
    return {
        **valid_response(),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I need to search",
                    "tool_calls": [tool_call()],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


@pytest.mark.parametrize("content", [None, "", "Searching"])
def test_gateway_accepts_complete_tool_response_without_repeating(content: str | None) -> None:
    calls = []
    output = tool_response()
    output["choices"][0]["message"]["content"] = content

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={**output, "debug": "private upstream state"})

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=tool_request())
    assert response.status_code == 200
    assert response.json() == output
    assert len(calls) == 1
    assert json.loads(calls[0].content) == tool_request()


def test_gateway_preserves_tool_history_and_reasoning_for_followup() -> None:
    payload = {
        **tool_request(),
        "messages": [
            {"role": "system", "content": "Use the search tool"},
            {"role": "user", "content": "Find the example"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need to search",
                "tool_calls": [tool_call()],
            },
            {"role": "tool", "tool_call_id": "call-search", "content": "Found the example"},
        ],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
    }
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert json.loads(calls[0].content) == payload


@pytest.mark.parametrize(
    "changes",
    [
        {"messages": [{"role": "assistant", "content": None}]},
        {"messages": [{"role": "assistant", "content": "", "tool_calls": []}]},
        {"messages": [{"role": "tool", "tool_call_id": "", "content": "result"}]},
        {"messages": [{"role": "tool", "tool_call_id": "call-search", "content": None}]},
        {"messages": [{"role": "user", "content": "input", "tool_calls": [tool_call()]}]},
        {"messages": [{"role": "user", "content": "input", "reasoning_content": "reasoning"}]},
        {"messages": [{"role": "assistant", "content": "input", "tool_call_id": "call-search"}]},
        {"tools": [{"type": "web_search"}]},
        {"tools": [{"type": "function", "function": {"name": "", "parameters": {}}}]},
        {"tools": [{"type": "function", "function": {"name": "search", "parameters": []}}]},
        {"tool_choice": {"type": "function", "function": {"name": ""}}},
        {"tool_choice": "other"},
        {"stream": 0},
        {"parallel_tool_calls": True},
    ],
)
def test_gateway_rejects_malformed_tool_requests(changes: dict[str, object]) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid tool requests must not reach the model")

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={**tool_request(), **changes})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "message",
    [
        {"content": None, "tool_calls": []},
        {"content": None, "tool_calls": [{"id": "x", "type": "function"}]},
        {"content": None, "tool_calls": [{**tool_call(), "id": ""}]},
        {
            "content": None,
            "tool_calls": [{**tool_call(), "function": {"name": "search", "arguments": {}}}],
        },
        {"content": None, "reasoning_content": "only thinking"},
    ],
)
def test_gateway_retries_malformed_tool_responses(message: dict[str, object]) -> None:
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                **valid_response(),
                "choices": [{"message": message, "finish_reason": "tool_calls"}],
            },
        )

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=tool_request())
    assert response.status_code == 502
    assert len(calls) == 6
    assert all(request.content == calls[0].content for request in calls)


@pytest.mark.parametrize(
    "parameters",
    [{"response_schema": {}}, {"trace_metadata": {}}],
)
def test_gateway_reserves_internal_request_fields(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", parameters=parameters
        )


def test_gateway_accepts_only_exact_echoes_of_fixed_parameters() -> None:
    fixed = {"reasoning_effort": "max", "temperature": 0.2, "extra": {"enabled": True}}
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model", parameters=fixed),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={**tool_request(), **fixed})
        assert response.status_code == 200
        for changes in (
            {"reasoning_effort": "high"},
            {"temperature": 1},
            {"extra": {"enabled": 1}},
            {"unknown": "max"},
        ):
            response = client.post(
                "/v1/chat/completions", json={**tool_request(), **fixed, **changes}
            )
            assert response.status_code == 400
    assert len(calls) == 1
    assert json.loads(calls[0].content) == {**tool_request(), **fixed}


def test_scoped_gateway_accounts_tool_cycles_and_retries(tmp_path: Path) -> None:
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json={**valid_response(), "choices": []})
        return httpx.Response(200, json=tool_response() if len(calls) == 2 else valid_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                f"/runs/{run_id}/{capability}/v1/chat/completions", json=tool_request()
            )
            assert response.status_code == 200
    assert read_usage(tmp_path, run_id) == {
        "prompt_tokens": 36,
        "completion_tokens": 9,
        "total_tokens": 45,
        "attempts": 3,
        "unreported_attempts": 0,
    }


def test_gateway_strips_nested_upstream_tool_metadata_without_retrying() -> None:
    output = tool_response()
    call = output["choices"][0]["message"]["tool_calls"][0]
    call["provider_specific_fields"] = {"private": "upstream metadata"}
    call["function"]["debug"] = "upstream metadata"
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=output)

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=tool_request())
    assert response.status_code == 200
    assert response.json() == tool_response()
    assert len(requests) == 1


def test_gateway_requires_usage_scope_for_tracing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", trace_dir=tmp_path
        )


def test_scoped_gateway_records_redacted_tool_attempts(tmp_path: Path) -> None:
    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage

    usage_dir = tmp_path / "usage"
    trace_dir = tmp_path / "traces"
    run_id = str(uuid4())
    capability = create_usage(usage_dir, run_id)
    calls = 0
    payload = tool_request()
    payload["messages"] = [
        {
            "role": "user",
            "content": f"http://gateway/runs/{run_id}/{capability}/v1 {capability} api-key-private",
        }
    ]

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("private endpoint api-key-private")
        return httpx.Response(200, json=tool_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            api_key="api-key-private",
            usage_dir=usage_dir,
            trace_dir=trace_dir,
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(f"/runs/{run_id}/{capability}/v1/chat/completions", json=payload)
    assert response.status_code == 200
    saved = read_calls(trace_dir, run_id)
    assert len(saved) == 2
    assert [call["status"] for call in saved] == ["ERROR", "OK"]
    assert [call["attributes"]["attempt"] for call in saved] == [1, 2]
    assert all(call["end_time_ns"] >= call["start_time_ns"] > 0 for call in saved)
    assert saved[1]["outputs"] == tool_response()
    encoded = json.dumps(saved)
    assert capability not in encoded
    assert "api-key-private" not in encoded
    assert "private endpoint" not in encoded
    assert "internal-model" not in encoded


def test_scoped_gateway_fails_when_trace_cannot_be_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluator.usage import create_usage, read_usage

    usage_dir = tmp_path / "usage"
    run_id = str(uuid4())
    capability = create_usage(usage_dir, run_id)

    def fail_trace(*args: object, **kwargs: object) -> None:
        raise OSError("private trace path")

    monkeypatch.setattr("evaluator.gateway.append_call", fail_trace)
    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            usage_dir=usage_dir,
            trace_dir=tmp_path / "traces",
        ),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=valid_response())),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body()
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "Model tracing unavailable"}
    assert read_usage(usage_dir, run_id)["total_tokens"] == 15


def vote_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string", "minLength": 1, "pattern": r"\S"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }


def test_gateway_retries_invalid_structured_outputs_with_one_retry_budget(tmp_path: Path) -> None:
    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage, read_usage

    usage_dir = tmp_path / "usage"
    trace_dir = tmp_path / "traces"
    run_id = str(uuid4())
    capability = create_usage(usage_dir, run_id)
    answers = [
        "not json",
        '{"score":true,"reason":"Reason"}',
        '{"score":1.0,"reason":"Reason"}',
        '{"score":6,"reason":"Reason"}',
        '{"score":5,"reason":"   "}',
        '{"score":5,"reason":"Reason"}',
    ]
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=valid_response(answers[len(requests) - 1]))

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            usage_dir=usage_dir,
            trace_dir=trace_dir,
            response_schemas=[vote_schema()],
            strict_schema_integers=True,
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions",
            json={**request_body(), "response_schema": vote_schema()},
        )
    assert response.status_code == 200
    assert response.json() == valid_response(answers[-1])
    assert len(requests) == 6
    assert all(request.content == requests[0].content for request in requests)
    assert json.loads(requests[0].content) == request_body()
    assert read_usage(usage_dir, run_id) == {
        "prompt_tokens": 72,
        "completion_tokens": 18,
        "total_tokens": 90,
        "attempts": 6,
        "unreported_attempts": 0,
    }
    traces = read_calls(trace_dir, run_id)
    assert [call["status"] for call in traces] == ["ERROR"] * 5 + ["OK"]
    assert traces[0]["outputs"] == valid_response(answers[0])


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://external.example/schema"},
        {"type": "integer", "mininum": 1},
        {"type": "invalid"},
        {"type": "string", "pattern": ".*"},
        [],
        True,
    ],
)
def test_gateway_rejects_invalid_output_schemas_before_model_call(schema: object) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid response schema must not reach model")

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={**request_body(), "response_schema": schema}
        )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid model request"}


def test_gateway_structured_output_exhaustion_stays_six_attempts() -> None:
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=valid_response('{"score":0,"reason":"Reason"}'))

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={**request_body(), "response_schema": vote_schema()}
        )
    assert response.status_code == 502
    assert len(requests) == 6


def test_gateway_validates_schema_only_for_final_tool_cycle_answer() -> None:
    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=tool_response())),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={**tool_request(), "response_schema": vote_schema()}
        )
    assert response.status_code == 200
    assert response.json() == tool_response()


def test_gateway_retries_unsolicited_tool_calls() -> None:
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=tool_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={**request_body(), "response_schema": vote_schema()}
        )
    assert response.status_code == 502
    assert len(requests) == 6


def test_gateway_returns_valid_low_score_without_repeating() -> None:
    requests = []
    output = valid_response('{"score":1,"reason":"Low quality"}')

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=output)

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={**request_body(), "response_schema": vote_schema()}
        )
    assert response.status_code == 200
    assert response.json() == output
    assert len(requests) == 1


def test_gateway_rejects_excessively_nested_request_json() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Excessive request nesting must not reach model")

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content="[" * 10000 + "0" + "]" * 10000)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid model request"}


def test_gateway_rejects_unapproved_output_schema_without_model_call() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unapproved response schema must not reach model")

    for approved in ([], [vote_schema()]):
        app = create_gateway(
            GatewaySettings(
                base_url="http://internal-model/v1", model="fixed-model", response_schemas=approved
            ),
            transport=httpx.MockTransport(upstream),
        )
        schema = vote_schema()
        if approved:
            schema = {
                **schema,
                "properties": {
                    "score": {"type": "integer", "const": 5},
                    "reason": {"type": "string"},
                },
            }
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions", json={**request_body(), "response_schema": schema}
            )
        assert response.status_code == 400


@pytest.mark.parametrize("schemas", [{}, [True], [{"$ref": "https://external.example/schema"}]])
def test_gateway_rejects_invalid_approved_schemas(schemas: Any) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", response_schemas=schemas
        )


def test_gateway_loads_approved_schemas_from_operator_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_BASE_URL", "http://internal-model/v1")
    monkeypatch.setenv("MODEL_NAME", "fixed-model")
    monkeypatch.setenv("MODEL_RESPONSE_SCHEMAS_JSON", json.dumps([vote_schema()]))
    assert GatewaySettings.from_env().response_schemas == [vote_schema()]


def test_gateway_trace_metadata_is_caller_context_and_not_an_upstream_parameter(
    tmp_path: Path,
) -> None:
    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage

    usage_dir = tmp_path / "usage"
    trace_dir = tmp_path / "traces"
    run_id = str(uuid4())
    capability = create_usage(usage_dir, run_id)
    metadata = {"role": "judge", "criterion": "precision", "vote": 1}
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            usage_dir=usage_dir,
            trace_dir=trace_dir,
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions",
            json={**request_body(), "trace_metadata": metadata},
        )
    assert response.status_code == 200
    assert json.loads(requests[0].content) == request_body()
    assert read_calls(trace_dir, run_id)[0]["attributes"]["caller_metadata"] == metadata


def test_gateway_rejects_nonfinite_trace_metadata() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Nonfinite trace metadata must not reach model")

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=json.dumps({**request_body(), "trace_metadata": {"invalid": float("nan")}}),
        )
    assert response.status_code == 400


@pytest.mark.parametrize("echo", [False, True])
def test_gateway_preserves_operator_fixed_tools(echo: bool) -> None:
    fixed = {key: value for key, value in tool_request().items() if key in {"tools", "tool_choice"}}
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=tool_response())

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model", parameters=fixed),
        transport=httpx.MockTransport(upstream),
    )
    body = {**request_body(), **(fixed if echo else {})}
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        for changed in (
            {"tools": None}, {"tools": []}, {"tool_choice": None}, {"tool_choice": "auto"}
        ):
            assert client.post("/v1/chat/completions", json={**body, **changed}).status_code == 400
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {**request_body(), **fixed}
    assert fixed == {key: tool_request()[key] for key in ("tools", "tool_choice")}


@pytest.mark.parametrize("fixed_field", ["tools", "tool_choice"])
def test_gateway_combines_operator_fixed_and_caller_tool_fields(fixed_field: str) -> None:
    body = tool_request()
    fixed = {fixed_field: body.pop(fixed_field)}
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=tool_response())

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model", parameters=fixed),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        invalid = (
            {"tool_choice": {"type": "function", "function": {"name": "other"}}}
            if fixed_field == "tools" else {"tools": None}
        )
        assert client.post("/v1/chat/completions", json={**body, **invalid}).status_code == 400
    assert len(requests) == 1
    assert json.loads(requests[0].content) == tool_request()


@pytest.mark.parametrize(
    "parameters",
    [{"tools": []}, {"tools": [{"type": "web_search"}]}, {"tool_choice": "invalid"}],
)
def test_gateway_rejects_invalid_operator_tool_fields(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", parameters=parameters
        )


def test_gateway_accepts_standard_integer_response_without_retry() -> None:
    requests = []
    output = valid_response('{"score":1.0,"reason":"Reason"}')

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=output)

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1",
            model="fixed-model",
            response_schemas=[vote_schema()],
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={**request_body(), "response_schema": vote_schema()}
        )
    assert response.status_code == 200
    assert response.json() == output
    assert len(requests) == 1


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_gateway_loads_integer_policy_from_operator_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("MODEL_BASE_URL", "http://internal-model/v1")
    monkeypatch.setenv("MODEL_NAME", "fixed-model")
    monkeypatch.setenv("MODEL_STRICT_SCHEMA_INTEGERS", value)
    assert GatewaySettings.from_env().strict_schema_integers is expected


@pytest.mark.parametrize("value", ["1", "null", '"true"', "TRUE"])
def test_gateway_rejects_invalid_integer_policy_environment(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MODEL_BASE_URL", "http://internal-model/v1")
    monkeypatch.setenv("MODEL_NAME", "fixed-model")
    monkeypatch.setenv("MODEL_STRICT_SCHEMA_INTEGERS", value)
    with pytest.raises(ValueError):
        GatewaySettings.from_env()


@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "search"}}])
def test_gateway_retries_with_effective_fixed_tool_configuration(choice: object) -> None:
    requests = []
    fixed = {"tool_choice": choice}
    body = request_body()
    if choice != "none":
        body["tools"] = tool_request()["tools"]

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=valid_response())

    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model", parameters=fixed),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert json.loads(requests[0].content) == {**body, **fixed}


def test_scoped_gateway_does_not_call_model_before_durable_attempt_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluator import usage as usage_store

    run_id = str(uuid4())
    capability = usage_store.create_usage(tmp_path, run_id)
    original_save = usage_store._save

    def fail_attempt_save(path, value):
        if value["attempts"]:
            raise OSError("synthetic storage failure")
        original_save(path, value)

    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json=valid_response())

    monkeypatch.setattr(usage_store, "_save", fail_attempt_save)
    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", usage_dir=tmp_path,
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body(),
        )
    assert response.status_code == 503
    assert calls == []
    assert usage_store.read_usage(tmp_path, run_id)["attempts"] == 0


@pytest.mark.parametrize("usage", [None, {"total_tokens": "private usage value"}])
def test_gateway_accepts_tool_call_with_unknown_usage_once(tmp_path: Path, usage: object) -> None:
    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={**tool_response(), "usage": usage})

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model",
            usage_dir=tmp_path, trace_dir=tmp_path / "traces",
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=tool_request(),
        )
    assert response.status_code == 200
    assert response.json() == {**tool_response(), "usage": None}
    assert len(requests) == 1
    assert read_usage(tmp_path, run_id)["unreported_attempts"] == 1
    trace = read_calls(tmp_path / "traces", run_id)[0]
    assert trace["status"] == "OK"
    assert trace["attributes"]["usage_reason"] == ("missing" if usage is None else "invalid")
    assert "private usage value" not in json.dumps(trace)


@pytest.mark.parametrize(
    "failure,reason,error_type,http_status",
    [
        (httpx.ConnectError("private endpoint"), "request", "ConnectError", None),
        (httpx.ReadTimeout("private endpoint"), "request", "ReadTimeout", None),
        (TimeoutError("private endpoint"), "request", "TimeoutError", None),
        (httpx.Response(200, content=b'{"choices":'), "response_decode", "JSONDecodeError", 200),
        (httpx.Response(503, content=b"private error body"), "http_status", "JSONDecodeError", 503),
        (httpx.Response(200, json={"choices": []}), "response_validation", "ValidationError", 200),
    ],
)
def test_gateway_records_safe_failure_diagnostics_and_attempt_ids(
    tmp_path: Path, failure: Exception | httpx.Response,
    reason: str, error_type: str, http_status: int | None,
) -> None:
    from uuid import UUID

    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    requests = []
    provider_id = str(uuid4())

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            if isinstance(failure, Exception):
                raise failure
            failure.headers["x-request-id"] = "private header value"
            return failure
        return httpx.Response(
            200, json=valid_response(), headers={"x-request-id": provider_id},
        )

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model",
            usage_dir=tmp_path, trace_dir=tmp_path / "traces",
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body(),
        )
    assert response.status_code == 200
    assert len(requests) == 2
    traces = read_calls(tmp_path / "traces", run_id)
    ids = [str(UUID(request.headers["x-request-id"])) for request in requests]
    assert len(set(ids)) == 2
    assert [trace["attributes"]["request_id"] for trace in traces] == ids
    failed, success = [trace["attributes"] for trace in traces]
    assert failed["error_reason"] == reason
    assert failed["exception_type"] == error_type
    assert failed.get("http_status") == http_status
    assert "upstream_request_id" not in failed
    assert success["upstream_request_id"] == provider_id
    assert success["usage_reason"] == "reported"
    assert "error_reason" not in success
    assert "private" not in json.dumps(traces)
    assert read_usage(tmp_path, run_id)["unreported_attempts"] == 1


def test_gateway_retains_http_error_usage_and_safe_provider_id(tmp_path: Path) -> None:
    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    requests = []
    provider_id = str(uuid4())

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            503 if len(requests) == 1 else 200, json=valid_response(),
            headers={"x-litellm-call-id": provider_id},
        )

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model",
            usage_dir=tmp_path, trace_dir=tmp_path / "traces",
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body(),
        )
    assert response.status_code == 200
    assert len(requests) == 2
    usage = read_usage(tmp_path, run_id)
    assert usage["total_tokens"] == 30
    assert usage["unreported_attempts"] == 0
    failed = read_calls(tmp_path / "traces", run_id)[0]["attributes"]
    assert failed["error_reason"] == "http_status"
    assert failed["http_status"] == 503
    assert failed["usage_reason"] == "reported"
    assert failed["upstream_request_id"] == provider_id


def test_gateway_records_cancellation_without_retrying(
    tmp_path: Path, retry_waits: list[float],
) -> None:
    from evaluator.trace_store import read_calls
    from evaluator.usage import create_usage, read_usage

    run_id = str(uuid4())
    capability = create_usage(tmp_path, run_id)
    calls = []

    async def scenario() -> None:
        started = asyncio.Event()

        async def upstream(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            started.set()
            await asyncio.Future()
            raise AssertionError("Cancelled request must not finish")

        app = create_gateway(
            GatewaySettings(
                base_url="http://internal-model/v1", model="fixed-model",
                usage_dir=tmp_path, trace_dir=tmp_path / "traces",
            ),
            transport=httpx.MockTransport(upstream),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway",
        ) as client:
            task = asyncio.create_task(client.post(
                f"/runs/{run_id}/{capability}/v1/chat/completions", json=request_body(),
            ))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
    assert len(calls) == 1
    assert retry_waits == []
    usage = read_usage(tmp_path, run_id)
    assert usage["attempts"] == usage["unreported_attempts"] == 1
    assert json.loads((tmp_path / f"{run_id}.json").read_text())["active_requests"] == 0
    traces = read_calls(tmp_path / "traces", run_id)
    assert len(traces) == 1
    assert traces[0]["status"] == "ERROR"
    assert traces[0]["attributes"]["error_reason"] == "cancelled"
    assert traces[0]["attributes"]["exception_type"] == "CancelledError"
