import json
import logging
import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "86fa9e6ca1de7ee71132d2be0103f3e37d1fa872"
CONFIG_BASE_URL = "http://arena-gateway.invalid/v1"
REQUEST_TIMEOUT = 131460
RUNNER_TIMEOUT = 604800


def gateway_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: message[key] for key in (
        "role", "content", "tool_calls", "tool_call_id", "reasoning_content"
    ) if key in message}
    if isinstance(result.get("tool_calls"), list):
        result["tool_calls"] = [{
            "id": call["id"],
            "type": call["type"],
            "function": {key: call["function"][key] for key in ("name", "arguments")},
        } for call in result["tool_calls"]]
    return result


def gateway_request(request: dict[str, Any]) -> dict[str, Any]:
    extra = request.get("extra_body")
    if isinstance(extra, dict):
        request = request | extra
    result = {
        key: request[key]
        for key in ("messages", "tools", "tool_choice")
        if key in request
    }
    if "messages" in result:
        result["messages"] = [gateway_message(message) for message in result["messages"]]
    result.update(model=MODEL, stream=False, reasoning_effort="max")
    return result


class ModelState:
    def __init__(self) -> None:
        self.failed = threading.Event()

    def check(self) -> None:
        if self.failed.is_set():
            raise RuntimeError("model gateway failed; further requests disabled")


def prepare_home(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        Path(__file__).with_name("config.yaml").read_text(), encoding="utf-8"
    )
    os.environ.update({
        "HERMES_HOME": str(home),
        "HERMES_SINGLE_QUERY_SESSION": "1",
        "OPENAI_BASE_URL": CONFIG_BASE_URL,
        "OPENAI_API_KEY": "arena-runtime-injects-credentials",
        "OPENAI_MODEL": MODEL,
        "HERMES_API_TIMEOUT": str(REQUEST_TIMEOUT),
        "HERMES_API_CALL_STALE_TIMEOUT": str(REQUEST_TIMEOUT),
        "TERMINAL_TIMEOUT": str(RUNNER_TIMEOUT),
        "TERMINAL_MAX_FOREGROUND_TIMEOUT": str(RUNNER_TIMEOUT),
        "TERMINAL_LIFETIME_SECONDS": str(RUNNER_TIMEOUT),
    })


def install(base_url: str, timeout: float, api_key: str = "arena") -> ModelState:
    logging.disable(logging.CRITICAL)
    import httpx
    import openai
    from openai.resources.chat.completions import AsyncCompletions, Completions

    state = ModelState()
    original_init = openai.OpenAI.__init__
    original_async_init = openai.AsyncOpenAI.__init__
    original_create = Completions.create
    original_async_create = AsyncCompletions.create

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        kwargs.update(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
        original_init(self, *args, **kwargs)

    @wraps(original_async_init)
    def initialize_async(self, *args, **kwargs):
        kwargs.update(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        kwargs["http_client"] = httpx.AsyncClient(trust_env=False, timeout=timeout)
        original_async_init(self, *args, **kwargs)

    @wraps(original_create)
    def create(self, *args, **kwargs):
        state.check()
        try:
            return original_create(self, *args, **gateway_request(kwargs))
        except Exception:
            state.failed.set()
            raise RuntimeError("model gateway request failed") from None

    @wraps(original_async_create)
    async def create_async(self, *args, **kwargs):
        state.check()
        try:
            return await original_async_create(self, *args, **gateway_request(kwargs))
        except Exception:
            state.failed.set()
            raise RuntimeError("model gateway request failed") from None

    openai.OpenAI.__init__ = initialize
    openai.AsyncOpenAI.__init__ = initialize_async
    Completions.create = create
    AsyncCompletions.create = create_async

    import tools.async_delegation as async_delegate
    import tools.delegate_tool as delegate
    from run_agent import AIAgent

    async_delegate._STALE_IDLE_SECONDS = RUNNER_TIMEOUT
    async_delegate._STALE_IN_TOOL_SECONDS = RUNNER_TIMEOUT
    delegate._HEARTBEAT_STALE_CYCLES_IDLE = RUNNER_TIMEOUT // 30
    delegate._HEARTBEAT_STALE_CYCLES_IN_TOOL = RUNNER_TIMEOUT // 30
    AIAgent._try_recover_primary_transport = lambda *args, **kwargs: False
    AIAgent._dump_api_request_debug = lambda *args, **kwargs: None
    return state


def credentials(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("model_base_url"), str):
        raise ValueError("invalid credentials")
    return data
