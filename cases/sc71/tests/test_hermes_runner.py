import json

import pytest

from cases.sc71.harness.hermes import main, mock_check, runtime


def task():
    return {
        "input": "Find the record",
        "model": runtime.MODEL,
        "model_base_url": "http://gateway/runs/id/secret/v1",
        "model_request_timeout": 130410,
        "prepare_timeout": 172800,
    }


class Registry:
    def __init__(self):
        self.tools = {}

    def register(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs


class Agent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.result = {"final_response": "answer", "completed": True, "failed": False}

    def run_conversation(self, question, system_message):
        self.question = question
        self.prompt = system_message
        return self.result

    def close(self):
        self.closed = True


@pytest.fixture
def candidate(tmp_path):
    path = tmp_path / "candidate"
    path.mkdir()
    (path / "prompt.md").write_text("Answer using evidence.")
    return path


def test_file_solver_preserves_fixed_model_and_files(candidate, tmp_path):
    registry = Registry()
    agents = []

    def factory(**kwargs):
        agent = Agent(**kwargs)
        agents.append(agent)
        return agent

    payload = task() | {"parameters": {"mode": "files"}}
    result = main.solve(payload, candidate, tmp_path, tmp_path / "work", factory, registry)
    assert result == "answer"
    agent = agents[0]
    assert agent.kwargs["model"] == runtime.MODEL
    assert agent.kwargs["base_url"] == runtime.CONFIG_BASE_URL
    assert agent.kwargs["reasoning_config"] == {"enabled": True, "effort": "max"}
    assert agent.kwargs["enabled_toolsets"] == ["terminal", "file", "delegation"]
    assert str(tmp_path) in agent.prompt
    assert agent.closed
    assert registry.tools == {}


def test_retriever_is_a_tool_and_must_be_called(candidate, tmp_path):
    (candidate / "retrieve.py").write_text(
        'import json, sys\npayload=json.load(sys.stdin)\n'
        'print(json.dumps({"results":[payload["arguments"]["query"]]}))\n'
    )
    registry = Registry()
    agents = []

    def factory(**kwargs):
        agent = Agent(**kwargs)
        agents.append(agent)
        return agent

    payload = task() | {"parameters": {"mode": "retrieval"}}
    with pytest.raises(ValueError, match="required retrieval tool"):
        main.solve(payload, candidate, tmp_path, tmp_path / "work", factory, registry)
    assert agents[0].closed
    assert "MUST call arena_search" in agents[0].prompt
    assert "arena_candidate" in agents[0].kwargs["enabled_toolsets"]

    class SearchingAgent(Agent):
        def run_conversation(self, question, system_message):
            result = registry.tools["arena_search"]["handler"]({"query": "evidence"})
            assert json.loads(result) == {"results": ["evidence"]}
            return self.result

    assert main.solve(
        payload, candidate, tmp_path, tmp_path / "work", SearchingAgent, registry
    ) == "answer"


def test_preparation_runs_inside_workspace(candidate, tmp_path):
    (candidate / "prepare.py").write_text(
        'import json, pathlib, sys\npayload=json.load(sys.stdin)\n'
        'pathlib.Path("index-created-by-method").write_text(payload["model"])\n'
        'print(json.dumps({"context":"Prepared context"}))\n'
    )
    agents = []

    def factory(**kwargs):
        agent = Agent(**kwargs)
        agents.append(agent)
        return agent

    workspace = tmp_path / "work"
    main.solve(task(), candidate, tmp_path, workspace, factory, Registry())
    assert (workspace / "index-created-by-method").read_text() == runtime.MODEL
    assert "Prepared context" in agents[0].prompt


@pytest.mark.parametrize("change", [
    {"model": "other"},
    {"uses_model": False},
    {"model_request_timeout": 0},
    {"prepare_timeout": float("nan")},
    {"parameters": {"mode": "unknown"}},
    {"parameters": {"prompt_file": "../outside"}},
    {"parameters": {"toolsets": ["web"]}},
])
def test_solver_rejects_invalid_task(candidate, tmp_path, change):
    with pytest.raises(ValueError):
        main.solve(task() | change, candidate, tmp_path, tmp_path / "work", Agent, Registry())


def test_failed_agent_never_becomes_a_valid_answer(candidate, tmp_path):
    class FailedAgent(Agent):
        def run_conversation(self, question, system_message):
            return {"failed": True, "completed": True, "final_response": "partial"}

    with pytest.raises(ValueError, match="Hermes failed"):
        main.solve(task(), candidate, tmp_path, tmp_path / "work", FailedAgent, Registry())


def test_gateway_request_preserves_tool_reasoning_replay():
    messages = [
        {"role": "assistant", "content": None, "reasoning_content": "reasoning",
         "tool_calls": [{"id": "call-1", "type": "function",
                         "function": {"name": "arena_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    source = {"model": "other", "messages": messages, "temperature": 0.2,
              "max_tokens": 500, "stream": True, "extra_body": {"bad": "value"}}
    assert runtime.gateway_request(source) == {
        "model": runtime.MODEL, "messages": messages,
        "stream": False, "reasoning_effort": "max",
    }


def test_tool_script_cannot_escape_candidate(candidate, tmp_path):
    (candidate / "tools.json").write_text(json.dumps([{
        "name": "escape", "description": "bad", "parameters": {}, "script": "../bad.py",
    }]))
    with pytest.raises(ValueError):
        main.solve(task(), candidate, tmp_path, tmp_path / "work", Agent, Registry())


def test_gateway_request_preserves_hermes_sdk_bypass():
    messages = [{"role": "user", "content": "Evidence query"}]
    tools = [{"type": "function", "function": {"name": "arena_search"}}]
    request = {
        "messages": [], "tools": [],
        "extra_body": {"messages": messages, "tools": tools, "model": "ignored"},
    }
    assert runtime.gateway_request(request) == {
        "model": runtime.MODEL, "messages": messages, "tools": tools,
        "stream": False, "reasoning_effort": "max",
    }


def test_trusted_build_gets_no_example_and_never_starts_agent(candidate, tmp_path):
    (candidate / "prompt.md").unlink()
    (candidate / "build.py").write_text(
        'import json,pathlib,sys\npayload=json.load(sys.stdin)\n'
        'assert "input" not in payload\n'
        'assert payload["parameters"]["method_option"] == "tree"\n'
        'assert payload["assets_dir"]\n'
        'assert payload["model_base_url"]\n'
        'pathlib.Path(payload["build_dir"],"generated.txt").write_text("synthetic")\n'
        'print(json.dumps({"status":"complete","details":"ignored"}))\n'
    )
    output = tmp_path / "output"
    output.mkdir()

    def forbidden_factory(**kwargs):
        raise AssertionError("build must not start Hermes")

    payload = {key: value for key, value in task().items() if key != "input"} | {
        "operation": "build",
        "parameters": {"method_option": "tree"},
    }
    result = main.solve(
        payload, candidate, tmp_path, tmp_path / "work", forbidden_factory, Registry(), output
    )
    assert result == "build complete"
    assert (output / "generated.txt").read_text() == "synthetic"


def test_trusted_build_uses_preparation_deadline(candidate, tmp_path, monkeypatch):
    (candidate / "build.py").write_text("raise AssertionError\n")
    captured = {}

    def run_script(path, payload, workspace, timeout):
        captured.update(path=path, payload=payload, timeout=timeout)
        return {}

    monkeypatch.setattr(main, "run_script", run_script)
    payload = {key: value for key, value in task().items() if key != "input"} | {
        "operation": "build",
    }
    assert main.solve(
        payload, candidate, tmp_path, tmp_path / "work", Agent, Registry()
    ) == "build complete"
    assert captured["path"] == candidate / "build.py"
    assert captured["timeout"] == payload["prepare_timeout"]
    assert captured["payload"]["build_dir"] == "/output"
    assert "input" not in captured["payload"]


@pytest.mark.parametrize("value", [1, True, "other", {}, None])
def test_operation_rejects_unknown_values(candidate, tmp_path, value):
    with pytest.raises(ValueError):
        main.solve(
            task() | {"operation": value},
            candidate, tmp_path, tmp_path / "work", Agent, Registry(),
        )


def test_gateway_request_removes_hermes_metadata_without_changing_history():
    history = [
        {"role": "assistant", "content": None, "reasoning": "private alias",
         "reasoning_content": "required replay", "timestamp": 1,
         "tool_calls": [{"id": "call-1", "type": "function", "index": 0,
                         "function": {"name": "arena_search", "arguments": "{}",
                                      "parsed_arguments": {}}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "evidence",
         "name": "arena_search", "tool_name": "arena_search", "timestamp": 2},
    ]
    expected = [
        {"role": "assistant", "content": None, "reasoning_content": "required replay",
         "tool_calls": [{"id": "call-1", "type": "function",
                         "function": {"name": "arena_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "evidence"},
    ]
    result = runtime.gateway_request({"messages": [], "extra_body": {"messages": history}})
    assert result["messages"] == expected
    assert history[0]["reasoning"] == "private alias"
    assert history[1]["name"] == "arena_search"


def test_build_rejects_example_input(candidate, tmp_path):
    with pytest.raises(ValueError, match="build cannot receive input"):
        main.solve(
            task() | {"operation": "build"}, candidate, tmp_path, tmp_path / "work",
            Agent, Registry(),
        )


def test_method_parameter_cannot_select_build(candidate, tmp_path):
    payload = task() | {"parameters": {"_arena_build": True}}
    assert main.solve(
        payload, candidate, tmp_path, tmp_path / "work", Agent, Registry()
    ) == "answer"


def test_explicit_solve_operation(candidate, tmp_path):
    assert main.solve(
        task() | {"operation": "solve"}, candidate, tmp_path, tmp_path / "work", Agent, Registry()
    ) == "answer"


def test_mock_gateway_finds_arena_virtualenv_after_move(monkeypatch):
    module_path = mock_check.Path("/opt/arena/cases/sc71/harness/hermes/mock_check.py")
    search_paths = list(mock_check.sys.path)
    monkeypatch.setattr(mock_check, "__file__", str(module_path))
    monkeypatch.setattr(mock_check.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(mock_check.sys, "path", search_paths)
    request_model, response_model = mock_check.gateway_models()
    version = f"python{mock_check.sys.version_info.major}.{mock_check.sys.version_info.minor}"
    assert search_paths[-1] == f"/opt/arena/.venv/lib/{version}/site-packages"
    assert request_model.__name__ == "ModelRequest"
    assert response_model.__name__ == "ModelResponse"
