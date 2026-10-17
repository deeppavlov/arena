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
    assert command == [sys.executable, "-I", "/candidate/solve.py"]
    assert json.loads(kwargs["input"]) == {
        "input": "task", "assets_dir": "/assets", "parameters": {"iterations": 3},
    }
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["check"] is True
    assert kwargs["stderr"] == subprocess.DEVNULL


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


@pytest.mark.parametrize("task", [
    {"input": "task"},
    {"input": "task", "uses_model": "false"},
    {"input": "task", "uses_model": 0},
    {"input": "task", "uses_model": False, "parameters": []},
    {"input": None, "uses_model": False},
    {"input": 0, "uses_model": False},
    {"uses_model": False},
    {"input": "", "model": "fixed", "model_base_url": "http://gateway/v1"},
])
def test_main_rejects_incomplete_or_invalid_mode(monkeypatch, capsys, task):
    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps(task).encode()
    )))
    monkeypatch.setattr(sample_runner, "solve", lambda task: "should not run")

    assert sample_runner.main() == 1
    assert capsys.readouterr().err == "runner failed\n"
