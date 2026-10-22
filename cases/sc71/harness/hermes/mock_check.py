import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cases.sc71.harness.hermes import runtime
from cases.sc71.harness.hermes.main import solve

SCENARIOS = ("retrieval", "files", "delegate", "nested", "failure")
FACT = "Synthetic fact: river amber has seven bridges."
ANSWER = "SYNTHETIC_FINAL seven bridges"
CHILD_GOAL = "CHILD_SYNTHETIC_MARKER"
CHILD_ANSWER = "CHILD_SYNTHETIC_RESULT seven bridges"
GRANDCHILD_GOAL = "GRANDCHILD_SYNTHETIC_MARKER"
GRANDCHILD_ANSWER = "GRANDCHILD_SYNTHETIC_RESULT seven bridges"
REASONING = "Synthetic reasoning."
SECRET = "cap-secret"


def gateway_models():
    if importlib.util.find_spec("jsonschema") is None:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        packages = Path(__file__).resolve().parents[4] / ".venv" / "lib" / version / "site-packages"
        sys.path.append(str(packages))
    from evaluator.gateway import ModelRequest, ModelResponse

    return ModelRequest, ModelResponse


class MockGateway:
    def __init__(self, scenario: str, document: Path) -> None:
        self.scenario = scenario
        self.document = document
        self.requests: list[dict[str, Any]] = []
        self.missing_tools: list[str] = []
        self.invalid_requests = 0
        self.child_seen = threading.Event()
        self.grandchild_seen = threading.Event()
        self.request_model, self.response_model = gateway_models()

    def respond(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.requests.append(payload)
        try:
            if payload.get("reasoning_effort") != "max":
                raise ValueError("Invalid model parameter")
            self.request_model.model_validate({key: value for key, value in payload.items()
                                               if key != "reasoning_effort"})
        except ValueError:
            self.invalid_requests += 1
            return 400, {"error": {"message": "Invalid model request"}}
        if self.scenario == "failure":
            return 500, {"error": {"message": "Synthetic gateway failure"}}
        messages = payload.get("messages", [])
        has_tool = any(message.get("role") == "tool" for message in messages)
        user_text = "\n".join(str(message.get("content", "")) for message in messages
                              if message.get("role") == "user")
        tool = None
        arguments: dict[str, Any] = {}
        answer = ANSWER
        if self.scenario == "retrieval" and not has_tool:
            tool, arguments = "arena_search", {"query": "river amber bridges"}
        if self.scenario == "files" and not has_tool:
            tool, arguments = "read_file", {"path": str(self.document)}
        if self.scenario in ("delegate", "nested"):
            if GRANDCHILD_GOAL in user_text:
                self.grandchild_seen.set()
                if not has_tool:
                    tool, arguments = "arena_search", {"query": "river amber bridges"}
                else:
                    answer = GRANDCHILD_ANSWER
            elif CHILD_GOAL in user_text:
                self.child_seen.set()
                if self.scenario == "nested" and not has_tool:
                    tool = "delegate_task"
                    goal = f"{GRANDCHILD_GOAL} Use arena_search, then return {GRANDCHILD_ANSWER}."
                    arguments = {"tasks": [{"goal": goal}]}
                else:
                    answer = CHILD_ANSWER
            elif not has_tool:
                tool = "delegate_task"
                arguments = {"tasks": [{"goal": f"{CHILD_GOAL} Return {CHILD_ANSWER}."}]}
            else:
                self.child_seen.wait(15)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if tool else answer,
            "reasoning_content": REASONING,
        }
        if tool:
            names = [(item.get("function") or {}).get("name") for item in payload.get("tools", [])]
            if tool not in names:
                self.missing_tools.append(tool)
            message["tool_calls"] = [{
                "id": f"call_synthetic_{len(self.requests)}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)},
            }]
        result = {
            "choices": [{
                "index": 0, "message": message,
                "finish_reason": "tool_calls" if tool else "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return 200, self.response_model.model_validate(result).model_dump(exclude_unset=True)

    def handler(self) -> type[BaseHTTPRequestHandler]:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def send_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self) -> None:
                self.send_json(200, {
                    "object": "list",
                    "data": [{"id": runtime.MODEL, "object": "model", "context_length": 128000}],
                })

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status, result = gateway.respond(payload)
                self.send_json(status, result)

        return Handler


def effective_settings() -> dict[str, bool]:
    from agent.tool_executor import (
        _resolve_concurrent_tool_timeout,
        _resolve_sequential_tool_timeout,
    )
    from tools import async_delegation, delegate_tool
    from tools.delegate_tool_config import (
        _get_max_concurrent_children,
        _get_max_spawn_depth,
        _get_oneshot_max_children,
    )

    heartbeat = delegate_tool._HEARTBEAT_INTERVAL
    return {
        "max_concurrent_children": _get_max_concurrent_children() == 30,
        "oneshot_max_children": _get_oneshot_max_children() == 30,
        "max_spawn_depth": _get_max_spawn_depth() == 3,
        "concurrent_tool_timeout": _resolve_concurrent_tool_timeout() == 604800,
        "sequential_tool_timeout": _resolve_sequential_tool_timeout() == 604800,
        "async_idle_timeout": async_delegation._STALE_IDLE_SECONDS == 604800,
        "async_tool_timeout": async_delegation._STALE_IN_TOOL_SECONDS == 604800,
        "child_idle_timeout": delegate_tool._HEARTBEAT_STALE_CYCLES_IDLE * heartbeat == 604800,
        "child_tool_timeout": delegate_tool._HEARTBEAT_STALE_CYCLES_IN_TOOL * heartbeat == 604800,
    }


def verify(
    gateway: MockGateway, answer: str | None, failed: bool, home: Path,
    agent_settings: dict[str, bool],
) -> dict[str, Any]:
    requests = gateway.requests
    messages = [message for request in requests for message in request.get("messages", [])]
    tool_results = [str(message.get("content", "")) for message in messages
                    if message.get("role") == "tool"]
    replays = [message for request in requests[1:] for message in request.get("messages", [])
               if message.get("role") == "assistant" and message.get("tool_calls")]
    replay_valid = bool(replays) and all(message.get("reasoning_content") == REASONING
                                         for message in replays)
    wire_valid = bool(requests) and all(
        request.get("model") == runtime.MODEL
        and request.get("reasoning_effort") == "max"
        and request.get("stream") is False
        and bool(request.get("messages"))
        for request in requests
    )
    secret_free = all(SECRET.encode() not in path.read_bytes()
                      for path in home.rglob("*") if path.is_file())
    failure = gateway.scenario == "failure"
    expected_result = CHILD_ANSWER if gateway.scenario == "delegate" else FACT
    tool_valid = any(expected_result in result for result in tool_results)
    checks = {
        **effective_settings(),
        **agent_settings,
        "strict_gateway_request": gateway.invalid_requests == 0,
        "fixed_model_parameters": wire_valid,
        "secret_free_home": secret_free,
        "request_count": len(requests) == {
            "failure": 1, "delegate": 3, "nested": 6,
        }.get(gateway.scenario, 2),
        "expected_outcome": answer is None and failed if failure else answer == ANSWER,
    }
    if not failure:
        checks.update(
            tool_advertised=not gateway.missing_tools,
            tool_result=tool_valid,
            reasoning_replay=replay_valid,
            gateway_healthy=not failed,
        )
    if gateway.scenario in ("delegate", "nested"):
        checks["child_completed"] = gateway.child_seen.is_set() and tool_valid
    if gateway.scenario == "nested":
        checks["grandchild_completed"] = gateway.grandchild_seen.is_set() and any(
            GRANDCHILD_ANSWER in result for result in tool_results
        )
    return {
        "scenario": gateway.scenario,
        "passed": all(checks.values()),
        "requests": len(requests),
        "checks": checks,
    }


def run_scenario(scenario: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="arena-hermes-mock-") as temporary:
        root = Path(temporary)
        candidate, assets, workspace, home = [root / name for name in
                                               ("candidate", "assets", "workspace", "home")]
        for path in (candidate, assets, workspace):
            path.mkdir()
        document = assets / "fact.md"
        document.write_text(FACT, encoding="utf-8")
        (candidate / "prompt.md").write_text(
            "Use the provided knowledge base tool and answer the synthetic task.", encoding="utf-8"
        )
        (candidate / "retrieve.py").write_text(
            "import json,sys\njson.load(sys.stdin)\n"
            f"print(json.dumps({{'documents': [{{'text': {FACT!r}}}]}}))\n", encoding="utf-8"
        )
        gateway = MockGateway(scenario, document)
        server = ThreadingHTTPServer(("127.0.0.1", 0), gateway.handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/runs/test/{SECRET}/v1"
            runtime.prepare_home(home)
            state = runtime.install(endpoint, runtime.REQUEST_TIMEOUT)
            from run_agent import AIAgent
            from tools.registry import registry

            task = {
                "input": "PARENT_SYNTHETIC_MARKER How many bridges does river amber have?",
                "model": runtime.MODEL,
                "uses_model": True,
                "model_base_url": endpoint,
                "model_request_timeout": runtime.REQUEST_TIMEOUT,
                "prepare_timeout": 172800,
                "parameters": {
                    "mode": "retrieval" if scenario in ("retrieval", "nested") else "files",
                    "toolsets": ["file", "delegation"]
                    if scenario in ("delegate", "nested") else ["file"],
                    "max_iterations": 8,
                },
            }
            answer = None
            agent_settings = {"api_timeout": False, "api_stale_timeout": False}

            def agent_factory(**kwargs):
                agent = AIAgent(**kwargs)
                agent_settings.update(
                    api_timeout=agent._resolved_api_call_timeout() == 130410,
                    api_stale_timeout=agent._resolved_api_call_stale_timeout_base()[0] == 130410,
                )
                return agent

            try:
                answer = solve(task, candidate, assets, workspace, agent_factory, registry)
                state.check()
            except Exception:
                pass
            result = verify(gateway, answer, state.failed.is_set(), home, agent_settings)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.scenario == "all":
            parser.error("worker requires one scenario")
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                try:
                    result = run_scenario(args.scenario)
                except Exception as error:
                    result = {
                        "scenario": args.scenario, "passed": False, "error": type(error).__name__,
                    }
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    results = []
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        try:
            process = subprocess.run(
                [sys.executable, "-m", "cases.sc71.harness.hermes.mock_check",
                 "--scenario", scenario, "--worker"],
                capture_output=True, text=True, timeout=180, check=False,
            )
            result = json.loads(process.stdout)
        except (subprocess.TimeoutExpired, ValueError):
            result = {"scenario": scenario, "passed": False, "error": "worker_failed"}
        results.append(result)
    passed = all(result["passed"] for result in results)
    print(json.dumps({"passed": passed, "hermes_revision": runtime.REVISION, "scenarios": results}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
