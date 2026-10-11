import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from evaluator.gateway import GatewaySettings, create_gateway


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
                    {"message": {"content": "Answer", "reasoning_content": "private reasoning"}}
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
        "choices": [{"message": {"content": "Answer"}}],
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
    assert calls == ["http://internal-model/v1/chat/completions"]


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": {"secret": "state"}}}]},
        {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": "secret"}},
    ],
)
def test_gateway_rejects_malformed_upstream_response(payload: dict[str, object]) -> None:
    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 502
    assert response.json() == {"detail": "Model unavailable"}


def test_gateway_limits_request_and_response_before_parsing() -> None:
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"x" * 257)

    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fixed-model", max_body_bytes=256
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
    assert len(calls) == 1


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
    app = create_gateway(
        GatewaySettings(base_url="http://internal-model/v1", model="fixed-model"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "complete"}, **finish_reason}]
                },
            )
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    assert response.json() == {"choices": [{"message": {"content": "complete"}, **finish_reason}]}


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
                    "choices": [{"message": {"content": "partial"}, "finish_reason": finish_reason}]
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
        return httpx.Response(200, json={"choices": [{"message": {"content": "complete"}}]})

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
