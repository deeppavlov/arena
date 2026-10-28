import base64
import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

import pytest

from cases.sc35.harness import hermes
from cases.sc71.harness.hermes import runtime


def task():
    document = BytesIO()
    with ZipFile(document, "w") as package:
        package.writestr("word/document.xml", "<document>Проверка 😀</document>")
    data = document.getvalue()
    return {
        "input": json.dumps({
            "case_id": "synthetic",
            "docx_base64": base64.b64encode(data).decode(),
            "document_sha256": hashlib.sha256(data).hexdigest(),
            "text_sha256": "b" * 64,
        }),
        "model": runtime.MODEL,
        "model_base_url": "http://gateway/runs/id/fixture-secret/v1",
        "model_request_timeout": 131460,
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

    def run_conversation(self, question, system_message):
        self.question = question
        self.prompt = system_message
        return {"completed": True, "failed": False, "final_response": '{"findings":[]}'}

    def close(self):
        self.closed = True


@pytest.fixture
def candidate(tmp_path):
    path = tmp_path / "candidate"
    path.mkdir()
    (path / "prompt.md").write_text("Inspect the current document.")
    return path


def test_document_is_a_file_and_base64_is_not_model_context(candidate, tmp_path):
    agents = []

    def factory(**kwargs):
        agent = Agent(**kwargs)
        agents.append(agent)
        return agent

    payload = task()
    item = json.loads(payload["input"])
    workspace = tmp_path / "work"
    result = hermes.solve(payload, candidate, tmp_path, workspace, factory, Registry())
    assert result == '{"findings":[]}'
    agent = agents[0]
    assert agent.closed
    assert agent.kwargs["model"] == runtime.MODEL
    assert agent.kwargs["reasoning_config"] == {"enabled": True, "effort": "max"}
    assert item["docx_base64"] not in agent.question + agent.prompt
    assert "fixture-secret" not in agent.question + agent.prompt
    assert "sc35-span-review-v1" in agent.prompt
    assert "knowledge base as answer evidence" not in agent.prompt
    assert json.loads(agent.question)["document_path"] == str(workspace / "input.docx")
    assert (workspace / "input.docx").read_bytes() == base64.b64decode(item["docx_base64"])
    assert json.loads((workspace / "input.json").read_text()) == item
    assert all(b"fixture-secret" not in path.read_bytes() for path in workspace.iterdir())


def test_preparation_and_tools_receive_original_envelope(candidate, tmp_path):
    script = (
        "import json,sys\np=json.load(sys.stdin)\n"
        "assert 'docx_base64' in json.loads(p['input'])\n"
        "assert p['model_base_url'].endswith('/fixture-secret/v1')\n"
        "print(json.dumps({'context':'synthetic context'}))\n"
    )
    (candidate / "prepare.py").write_text(script)
    (candidate / "inspect.py").write_text(script)
    (candidate / "tools.json").write_text(json.dumps([{
        "name": "inspect_document", "description": "Inspect current document", "parameters": {},
        "script": "inspect.py",
    }]))
    registry = Registry()

    class ToolAgent(Agent):
        def run_conversation(self, question, system_message):
            assert "synthetic context" in system_message
            assert "arena_candidate" in self.kwargs["enabled_toolsets"]
            result = registry.tools["inspect_document"]["handler"]({})
            assert json.loads(result) == {"context": "synthetic context"}
            return super().run_conversation(question, system_message)

    hermes.solve(task(), candidate, tmp_path, tmp_path / "work", ToolAgent, registry)


@pytest.mark.parametrize("change", [
    {"docx_base64": "not base64"},
    {"document_sha256": "a" * 64},
    {"case_id": "../outside"},
    {"text_sha256": "invalid"},
    {"reference": "closed"},
])
def test_invalid_document_is_rejected_before_agent(candidate, tmp_path, change):
    payload = task()
    payload["input"] = json.dumps(json.loads(payload["input"]) | change)

    def forbidden_factory(**kwargs):
        raise AssertionError("invalid input must not start Hermes")

    with pytest.raises(ValueError):
        hermes.solve(payload, candidate, tmp_path, tmp_path / "work", forbidden_factory, Registry())


def test_failed_agent_does_not_retry_or_return_partial_answer(candidate, tmp_path):
    calls = []

    class FailedAgent(Agent):
        def run_conversation(self, question, system_message):
            calls.append(question)
            return {"completed": True, "failed": True, "final_response": "partial"}

    with pytest.raises(ValueError, match="Hermes failed"):
        hermes.solve(task(), candidate, tmp_path, tmp_path / "work", FailedAgent, Registry())
    assert len(calls) == 1


def test_build_has_no_document_and_does_not_start_agent(candidate, tmp_path):
    (candidate / "build.py").write_text(
        "import json,pathlib,sys\np=json.load(sys.stdin)\nassert 'input' not in p\n"
        "pathlib.Path(p['build_dir'],'prepared.txt').write_text('synthetic')\nprint('{}')\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    payload = {key: value for key, value in task().items() if key != "input"}
    payload["operation"] = "build"

    def forbidden_factory(**kwargs):
        raise AssertionError("build must not start Hermes")

    result = hermes.solve(
        payload, candidate, tmp_path, tmp_path / "work", forbidden_factory, Registry(), output
    )
    assert result == "build complete"
    assert (output / "prepared.txt").read_text() == "synthetic"


def test_sc71_retrieval_mode_is_not_implicitly_enabled(candidate, tmp_path):
    payload = task() | {"parameters": {"mode": "retrieval"}}
    with pytest.raises(ValueError, match="SC35 uses document mode"):
        hermes.solve(payload, candidate, tmp_path, tmp_path / "work", Agent, Registry())
