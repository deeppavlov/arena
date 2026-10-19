import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runner import main as sample_runner


def test_program_harness_runs_frozen_script_without_model(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b'{"answer":"program output"}')

    def forbidden(*args, **kwargs):
        pytest.fail("model-free solve used model path")

    monkeypatch.setattr(sample_runner.subprocess, "run", run)
    monkeypatch.setattr(sample_runner, "read_limited", forbidden)
    monkeypatch.setattr(sample_runner.urllib.request, "build_opener", forbidden)
    result = sample_runner.solve({
        "input": "task", "uses_model": False, "parameters": {"iterations": 3},
        "prepare_timeout": 10,
    })

    assert result == "program output"
    command, kwargs = calls[0]
    assert command == [sys.executable, "-E", "-s", "/candidate/solve.py"]
    assert json.loads(kwargs["input"]) == {
        "input": "task", "assets_dir": "/assets", "parameters": {"iterations": 3},
    }
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["check"] is True
    assert kwargs["stderr"] == subprocess.DEVNULL


@pytest.mark.parametrize("filename, field", [("solve.py", "answer"), ("prepare.py", "context")])
def test_program_imports_siblings_and_packages_with_isolated_environment(
    tmp_path, monkeypatch, filename, field,
):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    package = candidate / "package"
    package.mkdir()
    (package / "__init__.py").write_text("from helper import transform\n")
    (candidate / "helper.py").write_text("def transform(value):\n    return value.upper()\n")
    script = candidate / filename
    script.write_text(
        "import importlib.util\nimport json\nimport sys\nfrom package import transform\n"
        "assert importlib.util.find_spec('outside_candidate') is None\n"
        "task = json.load(sys.stdin)\n"
        f"print(json.dumps({{{field!r}: transform(task['input'])}}))\n"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside_candidate.py").write_text("raise RuntimeError('must not import')\n")
    monkeypatch.setenv("PYTHONPATH", str(outside))
    monkeypatch.setenv("PYTHONHOME", str(outside))

    assert sample_runner.run_program(script, {"input": "sibling package"}, 10) == {
        field: "SIBLING PACKAGE",
    }


@pytest.mark.parametrize("raw", [
    b"[]", b"not JSON", b'{"answer":null}', b'{"answer":5}', b'{"context":"wrong"}',
    b"x" * (1024 * 1024 + 1),
], ids=["array", "not-json", "null", "number", "missing", "overflow"])
def test_program_harness_rejects_invalid_output(monkeypatch, raw):
    def run(command, **kwargs):
        kwargs["stdout"].write(raw)

    monkeypatch.setattr(sample_runner.subprocess, "run", run)

    with pytest.raises((ValueError, KeyError, TypeError)):
        sample_runner.solve({"input": "task", "uses_model": False, "prepare_timeout": 10})


@pytest.mark.parametrize("answer", ["", " \n\t", "answer"])
def test_program_harness_preserves_final_answer(monkeypatch, answer):
    def run(command, **kwargs):
        kwargs["stdout"].write(json.dumps({"answer": answer}).encode())

    monkeypatch.setattr(sample_runner.subprocess, "run", run)

    assert sample_runner.solve({
        "input": "task", "uses_model": False, "prepare_timeout": 10,
    }) == answer


@pytest.mark.parametrize("input_text", ["task", "", " \n\t"])
def test_main_accepts_program_task_without_model(monkeypatch, capsys, input_text):
    received = []

    def solve(task):
        received.append(task["input"])
        return "program output"

    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps({"input": input_text, "uses_model": False, "prepare_timeout": 10}).encode()
    )))
    monkeypatch.setattr(sample_runner, "solve", solve)

    assert sample_runner.main() == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "program output"}
    assert received == [input_text]


@pytest.mark.parametrize("input_text", ["task", "", " \n\t"])
def test_main_accepts_model_task_with_any_string_input(monkeypatch, capsys, input_text):
    received = []

    def solve(task):
        received.append(task["input"])
        return "model output"

    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps({
            "input": input_text, "model": "fixed", "model_base_url": "http://gateway/v1",
        }).encode()
    )))
    monkeypatch.setattr(sample_runner, "solve", solve)

    assert sample_runner.main() == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "model output"}
    assert received == [input_text]


@pytest.mark.parametrize("task", [
    {"input": "task"},
    {"input": "task", "uses_model": "false"},
    {"input": "task", "uses_model": 0},
    {"input": "task", "uses_model": False, "parameters": []},
    {"input": None, "uses_model": False},
    {"input": 0, "uses_model": False},
    {"uses_model": False},
    {"model": "fixed", "model_base_url": "http://gateway/v1"},
    {"input": None, "model": "fixed", "model_base_url": "http://gateway/v1"},
    {"input": 0, "model": "fixed", "model_base_url": "http://gateway/v1"},
    {"input": "", "model": " ", "model_base_url": "http://gateway/v1"},
    {"input": "", "model": "fixed", "model_base_url": " "},
])
def test_main_rejects_incomplete_or_invalid_mode(monkeypatch, capsys, task):
    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps(task).encode()
    )))
    monkeypatch.setattr(sample_runner, "solve", lambda task: "should not run")

    assert sample_runner.main() == 1
    assert capsys.readouterr().err == "runner failed\n"


@pytest.mark.parametrize("extra", [0, 1])
def test_main_task_input_limit_is_separate_from_output_limit(monkeypatch, capsys, extra):
    raw = json.dumps({"input": "task", "uses_model": False}).encode()
    raw += b" " * (8 * 1024 * 1024 - len(raw) + extra)
    received = []

    def solve(task):
        received.append(task)
        return "ok"

    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(sample_runner, "solve", solve)

    assert sample_runner.main() == extra
    output = capsys.readouterr()
    if extra:
        assert received == []
        assert output.out == ""
        assert output.err == "runner failed\n"
    else:
        assert received == [{"input": "task", "uses_model": False}]
        assert json.loads(output.out) == {"answer": "ok"}


def test_main_keeps_one_mebibyte_output_limit(monkeypatch, capsys):
    task = {"input": "task", "uses_model": False}
    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps(task).encode()
    )))
    monkeypatch.setattr(sample_runner, "solve", lambda task: "x" * (1024 * 1024))

    assert sample_runner.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "runner failed\n"


@pytest.mark.parametrize("filename, field", [("solve.py", "answer"), ("prepare.py", "context")])
def test_program_receives_large_unicode_input_and_surrogate_parameters(tmp_path, filename, field):
    payload = {"input": "😀" * 500000, "parameters": {"label": "я😀\ud800"}}
    script = tmp_path / filename
    script.write_text(
        "import hashlib,json,sys\n"
        "raw = sys.stdin.buffer.read()\n"
        "payload = json.loads(raw)\n"
        "digest = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()\n"
        f"print(json.dumps({{{field!r}: digest, 'bytes': len(raw)}}))\n"
    )

    result = sample_runner.run_program(script, payload, 10)

    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert result[field] == expected
    assert result["bytes"] < 2100000
