import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.runner import DockerRunner, RunnerError
from runner import main as sample_runner

IMAGE = "sha256:" + "a" * 64


def fake_docker(monkeypatch, script):
    commands = []
    cleanup = []
    popen = subprocess.Popen

    def start(command, **kwargs):
        commands.append(command)
        return popen([sys.executable, "-c", script], **kwargs)

    def remove(command, **kwargs):
        cleanup.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("evaluator.runner.subprocess.Popen", start)
    monkeypatch.setattr("evaluator.runner.subprocess.run", remove)
    return commands, cleanup


@pytest.mark.parametrize("image", ["python:3.12", "image:latest", "sha256:abc", "--help"])
def test_requires_pinned_operator_image(image):
    with pytest.raises(ValueError, match="pinned"):
        DockerRunner(image)


@pytest.mark.parametrize("network", ["host", "container:private", "", "--privileged"])
def test_rejects_shared_namespaces(network):
    with pytest.raises(ValueError, match="network"):
        DockerRunner(IMAGE, network=network)


def test_hardened_container_only_receives_task_and_public_mounts(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    script = (
        "import json,sys; task=json.load(sys.stdin); print(json.dumps({'answer':json.dumps(task)}))"
    )
    commands, cleanup = fake_docker(monkeypatch, script)
    runner = DockerRunner(IMAGE, network="arena-runners", model="internal-model", assets_dir=assets)

    result = json.loads(runner(candidate, "private task"))

    assert result == {
        "input": "private task",
        "model": "internal-model",
        "model_base_url": "http://model-gateway:8081/v1",
    }
    command = commands[0]
    assert command[:2] == ["docker", "run"]
    assert command[-1] == IMAGE
    for flag in ("--rm", "--read-only", "--interactive", "--init"):
        assert flag in command
    for flag, expected in {
        "--user": "10001:10001",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--pids-limit": "128",
        "--memory": "2g",
        "--cpus": "2",
        "--network": "arena-runners",
        "--log-driver": "none",
    }.items():
        assert command[command.index(flag) + 1] == expected
    mounts = [command[index + 1] for index, item in enumerate(command) if item == "--mount"]
    assert mounts == [
        f"type=bind,src={candidate.resolve()},dst=/candidate,readonly",
        f"type=bind,src={assets.resolve()},dst=/assets,readonly",
    ]
    assert not any("docker.sock" in item for item in command)
    assert "--privileged" not in command
    assert cleanup == []


def test_default_network_and_optional_assets(tmp_path, monkeypatch):
    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')

    assert DockerRunner(IMAGE)(tmp_path, "task") == "ok"

    command = commands[0]
    assert command[command.index("--network") + 1] == "none"
    assert command.count("--mount") == 1


@pytest.mark.parametrize("directory", ["missing", "bad,name"])
def test_rejects_invalid_mount_source(tmp_path, directory):
    candidate = tmp_path / directory
    if directory != "missing":
        candidate.mkdir()
    with pytest.raises(RunnerError, match="mount"):
        DockerRunner(IMAGE)(candidate, "task")


@pytest.mark.parametrize(
    "output",
    ["not json", '{"answer":3}', '{"answer":null}', "[]"],
)
def test_rejects_invalid_response_without_leaking_output(tmp_path, monkeypatch, output):
    fake_docker(monkeypatch, f"print({output!r})")

    with pytest.raises(RunnerError, match="^invalid runner response$"):
        DockerRunner(IMAGE)(tmp_path, "sensitive input")


@pytest.mark.parametrize("answer", ["", " \n\t"])
def test_empty_answer_is_returned_for_evaluator_scoring(tmp_path, monkeypatch, answer):
    output = json.dumps({"answer": answer})
    _, cleanup = fake_docker(monkeypatch, f"print({output!r})")

    assert DockerRunner(IMAGE)(tmp_path, "task") == answer
    assert cleanup == []


@pytest.mark.parametrize("answer", ["", " \n\t", "answer"])
def test_example_harness_preserves_answers_and_gateway_protocol(monkeypatch, answer):
    requests = []

    def open_request(request, timeout):
        requests.append(json.loads(request.data))
        response = {"choices": [{"message": {"content": answer}}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(sample_runner, "read_limited", lambda path: "candidate prompt")
    monkeypatch.setattr(Path, "is_file", lambda path: False)
    monkeypatch.setattr(
        sample_runner.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=open_request),
    )

    assert (
        sample_runner.solve(
            {"input": "task", "model": "internal", "model_base_url": "http://model-gateway:8081/v1"}
        )
        == answer
    )
    assert requests == [
        {
            "model": "internal",
            "messages": [
                {"role": "system", "content": "candidate prompt"},
                {"role": "user", "content": "task"},
            ],
        }
    ]


def test_failure_does_not_expose_task_or_stderr(tmp_path, monkeypatch):
    _, cleanup = fake_docker(
        monkeypatch,
        "import sys; print('private secret',file=sys.stderr); sys.exit(1)",
    )

    with pytest.raises(RunnerError, match="^runner failed$"):
        DockerRunner(IMAGE)(tmp_path, "sensitive input")

    assert cleanup[0][:3] == ["docker", "rm", "--force"]


def test_timeout_force_removes_container(tmp_path, monkeypatch):
    commands, cleanup = fake_docker(monkeypatch, "import time; time.sleep(30)")

    with pytest.raises(RunnerError, match="^runner timed out$"):
        DockerRunner(IMAGE, timeout=0.05)(tmp_path, "task")

    name = commands[0][commands[0].index("--name") + 1]
    assert cleanup == [["docker", "rm", "--force", name]]


def test_output_limit_force_removes_container(tmp_path, monkeypatch):
    _, cleanup = fake_docker(
        monkeypatch, "import sys; sys.stdout.write('x' * (1024 * 1024 + 1)); sys.stdout.flush()"
    )

    with pytest.raises(RunnerError, match="^runner output limit exceeded$"):
        DockerRunner(IMAGE)(tmp_path, "task")

    assert cleanup[0][:3] == ["docker", "rm", "--force"]


def test_missing_docker_is_sanitized(tmp_path, monkeypatch):
    def start(*args, **kwargs):
        raise FileNotFoundError("sensitive details")

    monkeypatch.setattr("evaluator.runner.subprocess.Popen", start)
    with pytest.raises(RunnerError, match="^runner unavailable$"):
        DockerRunner(IMAGE)(tmp_path, "task")


def test_deeply_nested_output_is_sanitized(tmp_path, monkeypatch):
    fake_docker(monkeypatch, "print('[' * 2000 + '0' + ']' * 2000)")

    with pytest.raises(RunnerError, match="^invalid runner response$"):
        DockerRunner(IMAGE)(tmp_path, "task")


def test_interrupted_evaluation_force_removes_container(tmp_path, monkeypatch):
    _, cleanup = fake_docker(monkeypatch, "import time; time.sleep(30)")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(DockerRunner, "_exchange", interrupt)
    with pytest.raises(KeyboardInterrupt):
        DockerRunner(IMAGE)(tmp_path, "task")

    assert cleanup[0][:3] == ["docker", "rm", "--force"]
