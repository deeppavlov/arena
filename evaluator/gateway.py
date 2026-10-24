import asyncio
import json
import os
from asyncio import sleep
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Literal, Self
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from evaluator.reliability import (
    MODEL_TIMEOUT,
    RETRY_DELAY,
    parse_usage,
    retry_delays,
    validate_timing,
)
from evaluator.response_schema import validate_response_content, validate_response_schema
from evaluator.trace_store import append_call
from evaluator.usage import add_usage, begin_request, end_request, ensure_open


@dataclass(frozen=True)
class GatewaySettings:
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    parameters: dict[str, object] = field(default_factory=dict)
    response_schemas: list[dict[str, object]] = field(default_factory=list)
    max_body_bytes: int = 16 * 1024 * 1024
    timeout_seconds: float = MODEL_TIMEOUT
    retry_delay: float = RETRY_DELAY
    usage_dir: Path | None = None
    trace_dir: Path | None = None
    max_response_bytes: int = 2 * 1024 * 1024
    strict_schema_integers: bool = False

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
        if self.trace_dir is not None and self.usage_dir is None:
            raise ValueError("Model tracing requires a usage scope")
        if not isinstance(self.parameters, dict):
            raise ValueError("MODEL_PARAMETERS_JSON must be an object")
        if self.parameters.keys() & {
            "model",
            "messages",
            "stream",
            "response_schema",
            "trace_metadata",
        }:
            raise ValueError("Reserved MODEL_PARAMETERS_JSON field")
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("MODEL_PARAMETERS_JSON must contain finite JSON values") from None
        if "tools" in self.parameters:
            tools = TypeAdapter(list[ToolDefinition] | None).validate_python(
                self.parameters["tools"], strict=True
            )
            if tools == []:
                raise ValueError("Operator tools must not be empty")
        if "tool_choice" in self.parameters:
            TypeAdapter(Literal["none", "auto", "required"] | ToolChoice | None).validate_python(
                self.parameters["tool_choice"], strict=True
            )
        if type(self.strict_schema_integers) is not bool:
            raise ValueError("MODEL_STRICT_SCHEMA_INTEGERS must be true or false")
        if not isinstance(self.response_schemas, list):
            raise ValueError("MODEL_RESPONSE_SCHEMAS_JSON must be a list")
        for schema in self.response_schemas:
            validate_response_schema(schema, strict_integers=self.strict_schema_integers)

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
            response_schemas=json.loads(os.environ.get("MODEL_RESPONSE_SCHEMAS_JSON", "[]")),
            strict_schema_integers=json.loads(
                os.environ.get("MODEL_STRICT_SCHEMA_INTEGERS", "false")
            ),
            timeout_seconds=float(os.environ.get("MODEL_TIMEOUT", str(MODEL_TIMEOUT))),
            retry_delay=float(os.environ.get("MODEL_RETRY_DELAY", str(RETRY_DELAY))),
            usage_dir=Path(os.environ["MODEL_USAGE_DIR"])
            if os.environ.get("MODEL_USAGE_DIR")
            else None,
            trace_dir=Path(os.environ["MODEL_TRACE_DIR"])
            if os.environ.get("MODEL_TRACE_DIR")
            else None,
        )


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: FunctionCall


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, min_length=1)
    tool_call_id: str | None = Field(default=None, min_length=1)
    reasoning_content: str | None = None

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if self.role == "assistant":
            if self.tool_call_id is not None or (self.content is None and not self.tool_calls):
                raise ValueError("Invalid assistant message")
        elif (
            self.content is None
            or self.tool_calls is not None
            or self.reasoning_content is not None
            or (self.role == "tool") != (self.tool_call_id is not None)
        ):
            raise ValueError("Invalid message fields for role")
        return self


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["function"]
    function: FunctionDefinition


class FunctionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)


class ToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["function"]
    function: FunctionChoice


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str
    messages: list[Message] = Field(min_length=1, max_length=1000)
    tools: list[ToolDefinition] | None = Field(default=None, min_length=1)
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None
    stream: bool = False
    response_schema: dict[str, JsonValue] | None = None
    trace_metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        if self.stream:
            raise ValueError("Streaming is not supported")
        names = {tool.function.name for tool in self.tools or []}
        if self.tool_choice not in (None, "none") and not names:
            raise ValueError("Tool choice requires tools")
        if isinstance(self.tool_choice, ToolChoice) and self.tool_choice.function.name not in names:
            raise ValueError("Unknown selected tool")
        return self


class ResponseFunctionCall(FunctionCall):
    model_config = ConfigDict(extra="ignore")


class ResponseToolCall(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: ResponseFunctionCall


class ResponseMessage(BaseModel):
    model_config = ConfigDict(strict=True)
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ResponseToolCall] | None = Field(default=None, min_length=1)
    reasoning_content: str | None = None


class Choice(BaseModel):
    model_config = ConfigDict(strict=True)
    index: int = Field(default=0, ge=0)
    message: ResponseMessage
    finish_reason: Literal["stop", "tool_calls"] = "stop"

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if self.finish_reason == "tool_calls":
            if not self.message.tool_calls:
                raise ValueError("Missing tool calls")
        elif self.message.tool_calls or not (self.message.content or "").strip():
            raise ValueError("Invalid final response")
        return self


class Usage(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    choices: list[Choice] = Field(min_length=1, max_length=16)
    usage: Usage | None


def _exception_type(error: Exception) -> str:
    name = type(error).__name__
    if name in {
        "ConnectError", "ReadError", "WriteError", "CloseError", "ConnectTimeout",
        "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError",
        "LocalProtocolError", "ProxyError", "UnsupportedProtocol", "DecodingError",
        "TooManyRedirects", "HTTPStatusError", "TimeoutError", "JSONDecodeError",
        "UnicodeDecodeError", "ValidationError", "ValueError",
    }:
        return name
    return "HTTPError" if isinstance(error, httpx.HTTPError) else "ValueError"


def _upstream_request_id(headers: httpx.Headers) -> str | None:
    for name in ("x-request-id", "x-litellm-call-id"):
        value = headers.get(name, "")
        try:
            if len(value) == 36 and str(UUID(value)) == value.lower():
                return value
        except ValueError:
            pass
    return None


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
            return await complete(request, run_id, capability)
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

    async def complete(
        request: Request, run_id: str | None, capability: str | None = None
    ) -> dict[str, object]:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > settings.max_body_bytes:
                raise HTTPException(status_code=413, detail="Model request too large")
            body.extend(chunk)
        try:
            decoded_request = json.loads(body)
            if not isinstance(decoded_request, dict):
                raise ValueError("Invalid model request")
            for key in decoded_request.keys() & settings.parameters.keys():
                if json.dumps(decoded_request[key], sort_keys=True, allow_nan=False) != json.dumps(
                    settings.parameters[key], sort_keys=True, allow_nan=False
                ):
                    raise ValueError("Invalid model parameter")
            for key in decoded_request.keys() - ModelRequest.model_fields.keys():
                if key not in settings.parameters:
                    raise ValueError("Invalid model parameter")
                del decoded_request[key]
            for key in ("tools", "tool_choice"):
                if key in settings.parameters:
                    decoded_request[key] = settings.parameters[key]
            parsed = ModelRequest.model_validate(decoded_request)
            if parsed.response_schema is not None and not any(
                json.dumps(parsed.response_schema, sort_keys=True, allow_nan=False)
                == json.dumps(schema, sort_keys=True, allow_nan=False)
                for schema in settings.response_schemas
            ):
                raise ValueError("Unapproved response schema")
            schema_validator = (
                validate_response_schema(
                    parsed.response_schema, strict_integers=settings.strict_schema_integers
                )
                if parsed.response_schema is not None
                else None
            )
        except (ValueError, RecursionError):
            raise HTTPException(status_code=400, detail="Invalid model request") from None
        if parsed.model != settings.model or request.url.query:
            raise HTTPException(status_code=400, detail="Invalid model request")
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        payload = {
            **settings.parameters,
            **parsed.model_dump(exclude_unset=True, exclude={"response_schema", "trace_metadata"}),
        }
        delays = retry_delays(retry_delay=settings.retry_delay)
        for attempt in range(len(delays) + 1):
            ensure_running(run_id)
            if settings.usage_dir is not None and run_id is not None:
                try:
                    add_usage(settings.usage_dir, run_id, None)
                except (OSError, ValueError):
                    raise HTTPException(
                        status_code=503, detail="Model accounting unavailable"
                    ) from None
            usage = None
            started = time_ns()
            traced_output: dict[str, object] = {"error": "Model unavailable"}
            status = "ERROR"
            request_id = str(uuid4())
            diagnostics: dict[str, object] = {
                "request_id": request_id, "usage_reason": "response_unavailable",
            }
            error_reason = "request"
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
                            headers={**headers, "X-Request-ID": request_id},
                            json=payload,
                        ) as response:
                            diagnostics["http_status"] = response.status_code
                            upstream_id = _upstream_request_id(response.headers)
                            if upstream_id is not None:
                                diagnostics["upstream_request_id"] = upstream_id
                            error_reason = (
                                "response_read" if response.status_code == 200 else "http_status"
                            )
                            output = bytearray()
                            async for chunk in response.aiter_bytes():
                                if len(output) + len(chunk) > settings.max_response_bytes:
                                    error_reason = "response_too_large"
                                    raise ValueError("Upstream response too large")
                                output.extend(chunk)
                diagnostics["usage_reason"] = "unreadable_response"
                if response.status_code == 200:
                    error_reason = "response_decode"
                decoded = json.loads(output)
                if not isinstance(decoded, dict):
                    if response.status_code == 200:
                        error_reason = "response_shape"
                    raise ValueError("Invalid model response")
                diagnostics["usage_reason"] = (
                    "missing" if decoded.get("usage") is None else "invalid"
                )
                try:
                    usage = parse_usage(decoded.get("usage"))
                except ValueError:
                    pass
                else:
                    diagnostics["usage_reason"] = "reported"
                if response.status_code != 200:
                    raise ValueError("Upstream failure")
                error_reason = "response_validation"
                result = ModelResponse.model_validate({**decoded, "usage": usage})
                traced_output = result.model_dump(exclude_unset=True)
                error_reason = "response_schema"
                for choice in result.choices:
                    if choice.finish_reason == "tool_calls":
                        if not parsed.tools:
                            raise ValueError("Unexpected tool calls")
                    elif schema_validator is not None:
                        validate_response_content(choice.message.content or "", schema_validator)
                status = "OK"
                return traced_output
            except asyncio.CancelledError:
                diagnostics.update(error_reason="cancelled", exception_type="CancelledError")
                raise
            except (httpx.HTTPError, TimeoutError, ValueError) as error:
                diagnostics.update(
                    error_reason=error_reason, exception_type=_exception_type(error),
                )
                if attempt == len(delays):
                    raise HTTPException(status_code=502, detail="Model unavailable") from None
            finally:
                if settings.usage_dir is not None and run_id is not None:
                    try:
                        add_usage(settings.usage_dir, run_id, usage, reserved=True)
                    except (OSError, ValueError):
                        raise HTTPException(
                            status_code=503, detail="Model accounting unavailable"
                        ) from None
                if settings.trace_dir is not None and run_id is not None:
                    try:
                        append_call(
                            settings.trace_dir,
                            run_id,
                            {
                                "name": "model.chat.completions",
                                "span_type": "LLM",
                                "inputs": payload,
                                "outputs": traced_output,
                                "attributes": {
                                    "model": settings.model,
                                    "attempt": attempt + 1,
                                    "usage": usage,
                                    "caller_metadata": parsed.trace_metadata,
                                    **diagnostics,
                                },
                                "status": status,
                                "start_time_ns": started,
                                "end_time_ns": time_ns(),
                            },
                            secrets=tuple(
                                value for value in (settings.api_key, capability) if value
                            ),
                        )
                    except (OSError, ValueError):
                        raise HTTPException(
                            status_code=503, detail="Model tracing unavailable"
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
