import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runner import main as sample_runner


def invoke(monkeypatch, task):
    monkeypatch.setattr(sample_runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps(task).encode()
    )))
    return sample_runner.main()


@pytest.mark.parametrize("uses_model", [False, True])
def test_build_dispatches_script_without_prompt_or_final_model_request(
    monkeypatch, capsys, uses_model,
):
    calls = []

    def run_program(path, payload, timeout):
        calls.append((path, payload, timeout))
        return {"description": "prepared files"}

    def forbidden(*args, **kwargs):
        pytest.fail("build executed solve or a final model request")

    monkeypatch.setattr(sample_runner, "run_program", run_program)
    monkeypatch.setattr(sample_runner, "solve", forbidden)
    monkeypatch.setattr(sample_runner, "read_limited", forbidden)
    monkeypatch.setattr(sample_runner.urllib.request, "build_opener", forbidden)
    task = {
        "operation": "build", "uses_model": uses_model,
        "prepare_timeout": 19, "parameters": {"format": "table"},
        "model": "fixed", "model_base_url": "http://gateway/v1",
        "model_request_timeout": 7,
    }

    assert invoke(monkeypatch, task) == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "build complete"}
    expected = {
        "assets_dir": "/assets", "build_dir": "/output", "uses_model": uses_model,
        "parameters": {"format": "table"},
    }
    if uses_model:
        expected.update({
            "model": "fixed", "model_base_url": "http://gateway/v1",
            "model_request_timeout": 7,
        })
    assert calls == [(Path("/candidate/build.py"), expected, 19)]


@pytest.mark.parametrize("change", [
    {"operation": "unknown"},
    {"operation": None},
    {"operation": []},
    {"operation": "build", "input": "question"},
    {"operation": "build", "input": ""},
    {"operation": "build", "input": None},
    {"operation": "build", "input": " "},
])
def test_invalid_operation_rejected_before_execution(monkeypatch, capsys, change):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid operation executed")

    monkeypatch.setattr(sample_runner, "solve", forbidden)
    monkeypatch.setattr(sample_runner, "run_program", forbidden)

    assert invoke(monkeypatch, {
        "operation": "build", "uses_model": False, "prepare_timeout": 10,
    } | change) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "runner failed\n"


def test_real_program_build_writes_output_without_running_solver(tmp_path, monkeypatch, capsys):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (candidate / "build.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "assert set(payload) == {'assets_dir', 'build_dir', 'uses_model', 'parameters'}\n"
        "assert payload['uses_model'] is False\n"
        "Path(payload['parameters']['test_output']).joinpath('table.txt').write_text('1,2,3')\n"
        "print('{}')\n"
    )
    (candidate / "solve.py").write_text("raise RuntimeError('solver must not run')\n")
    monkeypatch.setattr(sample_runner, "Path", lambda path: candidate / Path(path).name)

    assert invoke(monkeypatch, {
        "operation": "build", "uses_model": False,
        "prepare_timeout": 10, "parameters": {"test_output": str(output)},
    }) == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "build complete"}
    assert (output / "table.txt").read_text() == "1,2,3"


@pytest.mark.parametrize("script", [None, "print('[]')\n", "raise RuntimeError('failed')\n"])
def test_build_failure_does_not_fall_back_to_solver(tmp_path, monkeypatch, capsys, script):
    if script is not None:
        (tmp_path / "build.py").write_text(script)
    (tmp_path / "solve.py").write_text("print('{\"answer\":\"incorrect success\"}')\n")
    monkeypatch.setattr(sample_runner, "Path", lambda path: tmp_path / Path(path).name)

    assert invoke(monkeypatch, {
        "operation": "build", "uses_model": False, "prepare_timeout": 10,
    }) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "runner failed\n"
