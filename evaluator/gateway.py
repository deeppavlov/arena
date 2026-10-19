import asyncio
import json
import os
from asyncio import sleep
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evaluator.reliability import (
    MODEL_TIMEOUT,
    RETRY_DELAY,
    parse_usage,
    retry_delays,
    validate_timing,
)
from evaluator.usage import add_usage, begin_request, end_request, ensure_open


@dataclass(frozen=True)
class GatewaySettings:
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    parameters: dict[str, object] = field(default_factory=dict)
    max_body_bytes: int = 16 * 1024 * 1024
    timeout_seconds: float = MODEL_TIMEOUT
    retry_delay: float = RETRY_DELAY
    usage_dir: Path | None = None
    max_response_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or not url.path.rstrip("/").endswith("/v1")
        ):
            raise ValueError("MODEL_BASE_URL must be an internal HTTP(S) /v1 endpoint")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.model.strip() or min(self.max_body_bytes, self.max_response_bytes) < 1:
            raise ValueError("Invalid model gateway configuration")
        object.__setattr__(
            self, "timeout_seconds", validate_timing(self.timeout_seconds, "timeout")
        )
        object.__setattr__(
            self, "retry_delay", validate_timing(self.retry_delay, "retry_delay", allow_zero=True)
        )
        if not isinstance(self.parameters, dict):
            raise ValueError("MODEL_PARAMETERS_JSON must be an object")
        if self.parameters.keys() & {"model", "messages", "stream"}:
            raise ValueError("Reserved MODEL_PARAMETERS_JSON field")
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("MODEL_PARAMETERS_JSON must contain finite JSON values") from None

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        key_file = os.environ.get("MODEL_API_KEY_FILE")
        parameters = json.loads(os.environ.get("MODEL_PARAMETERS_JSON", "{}"))
        if not isinstance(parameters, dict):
            raise ValueError("MODEL_PARAMETERS_JSON must be an object")
        return cls(
            base_url=os.environ["MODEL_BASE_URL"],
            model=os.environ["MODEL_NAME"],
            api_key=Path(key_file).read_text(encoding="utf-8").strip() if key_file else None,
            parameters=parameters,
            timeout_seconds=float(os.environ.get("MODEL_TIMEOUT", str(MODEL_TIMEOUT))),
            retry_delay=float(os.environ.get("MODEL_RETRY_DELAY", str(RETRY_DELAY))),
            usage_dir=Path(os.environ["MODEL_USAGE_DIR"])
            if os.environ.get("MODEL_USAGE_DIR")
            else None,
        )


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    messages: list[Message] = Field(min_length=1, max_length=1000)


class ResponseMessage(BaseModel):
    model_config = ConfigDict(strict=True)
    content: str


class Choice(BaseModel):
    message: ResponseMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    choices: list[Choice] = Field(min_length=1, max_length=16)
    usage: Usage


def create_gateway(
    settings: GatewaySettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> dict[str, object]:
        if settings.usage_dir is not None:
            raise HTTPException(status_code=403, detail="Model usage scope required")
        return await complete(request, None)

    @app.post("/runs/{run_id}/{capability}/v1/chat/completions")
    async def scoped_completions(
        request: Request, run_id: str, capability: str
    ) -> dict[str, object]:
        if settings.usage_dir is None:
            raise HTTPException(status_code=404, detail="Unknown model usage scope")
        try:
            begin_request(settings.usage_dir, run_id, capability)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="Unknown model usage scope") from None
        except OSError:
            raise HTTPException(status_code=503, detail="Model accounting unavailable") from None
        try:
            return await complete(request, run_id)
        finally:
            try:
                end_request(settings.usage_dir, run_id)
            except (OSError, ValueError):
                raise HTTPException(
                    status_code=503, detail="Model accounting unavailable"
                ) from None

    def ensure_running(run_id: str | None) -> None:
        if settings.usage_dir is not None and run_id is not None:
            try:
                ensure_open(settings.usage_dir, run_id)
            except (OSError, ValueError):
                raise HTTPException(status_code=502, detail="Model unavailable") from None

    async def complete(request: Request, run_id: str | None) -> dict[str, object]:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > settings.max_body_bytes:
                raise HTTPException(status_code=413, detail="Model request too large")
            body.extend(chunk)
        try:
            parsed = ModelRequest.model_validate_json(body)
        except ValidationError:
            raise HTTPException(status_code=400, detail="Invalid model request") from None
        if parsed.model != settings.model or request.url.query:
            raise HTTPException(status_code=400, detail="Invalid model request")
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        payload = {
            **settings.parameters,
            "model": settings.model,
            "messages": [message.model_dump() for message in parsed.messages],
        }
        delays = retry_delays(retry_delay=settings.retry_delay)
        for attempt in range(len(delays) + 1):
            ensure_running(run_id)
            usage = None
            try:
                async with asyncio.timeout(settings.timeout_seconds):
                    async with httpx.AsyncClient(
                        transport=transport,
                        timeout=settings.timeout_seconds,
                        follow_redirects=False,
                        trust_env=False,
                    ) as client:
                        async with client.stream(
                            "POST",
                            f"{settings.base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                        ) as response:
                            output = bytearray()
                            async for chunk in response.aiter_bytes():
                                if len(output) + len(chunk) > settings.max_response_bytes:
                                    raise ValueError("Upstream response too large")
                                output.extend(chunk)
                decoded = json.loads(output)
                if not isinstance(decoded, dict):
                    raise ValueError("Invalid model response")
                usage = parse_usage(decoded.get("usage"))
                if response.status_code != 200:
                    raise ValueError("Upstream failure")
                result = ModelResponse.model_validate(decoded)
                if any(not choice.message.content.strip() for choice in result.choices):
                    raise ValueError("Empty model response")
                return result.model_dump(exclude_none=True, exclude_unset=True)
            except (httpx.HTTPError, TimeoutError, ValueError):
                if attempt == len(delays):
                    raise HTTPException(status_code=502, detail="Model unavailable") from None
            finally:
                if settings.usage_dir is not None and run_id is not None:
                    try:
                        add_usage(settings.usage_dir, run_id, usage)
                    except (OSError, ValueError):
                        raise HTTPException(
                            status_code=503, detail="Model accounting unavailable"
                        ) from None
            ensure_running(run_id)
            await sleep(delays[attempt])
        raise HTTPException(status_code=502, detail="Model unavailable")

    return app


def main() -> None:
    uvicorn.run(
        create_gateway(GatewaySettings.from_env()), host="0.0.0.0", port=8081, access_log=False
    )


if __name__ == "__main__":
    main()
