import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from evaluator.runner import DockerRunner, RunnerError, SolveResult
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
    runner = DockerRunner(
        IMAGE,
        network="arena-runners",
        model="internal-model",
        assets_dir=assets,
        usage_dir=tmp_path / "usage",
    )

    result = json.loads(runner(candidate, "private task").answer)
    scoped_url = result.pop("model_base_url")
    run_id = scoped_url.split("/runs/")[1].split("/")[0]
    assert str(UUID(run_id)) == run_id
    assert scoped_url == f"http://model-gateway:8081/runs/{run_id}/v1"

    assert result == {
        "input": "private task",
        "model": "internal-model",
        "model_request_timeout": 130410,
        "prepare_timeout": 172800,
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

    assert DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task").answer == "ok"

    command = commands[0]
    assert command[command.index("--network") + 1] == "none"
    assert command.count("--mount") == 1


@pytest.mark.parametrize("directory", ["missing", "bad,name"])
def test_rejects_invalid_mount_source(tmp_path, directory):
    candidate = tmp_path / directory
    if directory != "missing":
        candidate.mkdir()
    with pytest.raises(RunnerError, match="mount"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(candidate, "task")


@pytest.mark.parametrize(
    "output",
    ["not json", '{"answer":3}', '{"answer":null}', "[]"],
)
def test_rejects_invalid_response_without_leaking_output(tmp_path, monkeypatch, output):
    fake_docker(monkeypatch, f"print({output!r})")

    with pytest.raises(RunnerError, match="^invalid runner response$"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "sensitive input")


@pytest.mark.parametrize("answer", ["", " \n\t"])
def test_empty_answer_is_returned_for_evaluator_scoring(tmp_path, monkeypatch, answer):
    output = json.dumps({"answer": answer})
    _, cleanup = fake_docker(monkeypatch, f"print({output!r})")

    assert DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task").answer == answer
    assert cleanup == []


@pytest.mark.parametrize("answer", ["", " \n\t", "answer"])
def test_example_harness_preserves_answers_and_gateway_protocol(monkeypatch, answer):
    requests = []

    def open_request(request, timeout):
        requests.append(json.loads(request.data))
        assert timeout == 130410
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
            {
                "input": "task",
                "model": "internal",
                "model_base_url": "http://model-gateway:8081/runs/example/v1",
                "model_request_timeout": 130410,
                "prepare_timeout": 172800,
            }
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
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "sensitive input")

    assert cleanup[0][:3] == ["docker", "rm", "--force"]


def test_timeout_force_removes_container(tmp_path, monkeypatch):
    commands, cleanup = fake_docker(monkeypatch, "import time; time.sleep(30)")

    with pytest.raises(RunnerError, match="^runner timed out$"):
        DockerRunner(IMAGE, timeout=0.05, usage_dir=tmp_path / "usage")(tmp_path, "task")

    name = commands[0][commands[0].index("--name") + 1]
    assert cleanup == [["docker", "rm", "--force", name]]


def test_output_limit_force_removes_container(tmp_path, monkeypatch):
    _, cleanup = fake_docker(
        monkeypatch, "import sys; sys.stdout.write('x' * (1024 * 1024 + 1)); sys.stdout.flush()"
    )

    with pytest.raises(RunnerError, match="^runner output limit exceeded$"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")

    assert cleanup[0][:3] == ["docker", "rm", "--force"]


def test_missing_docker_is_sanitized(tmp_path, monkeypatch):
    def start(*args, **kwargs):
        raise FileNotFoundError("sensitive details")

    monkeypatch.setattr("evaluator.runner.subprocess.Popen", start)
    with pytest.raises(RunnerError, match="^runner unavailable$"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")


def test_deeply_nested_output_is_sanitized(tmp_path, monkeypatch):
    fake_docker(monkeypatch, "print('[' * 2000 + '0' + ']' * 2000)")

    with pytest.raises(RunnerError, match="^invalid runner response$"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")


def test_interrupted_evaluation_force_removes_container(tmp_path, monkeypatch):
    _, cleanup = fake_docker(monkeypatch, "import time; time.sleep(30)")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(DockerRunner, "_exchange", interrupt)
    with pytest.raises(KeyboardInterrupt):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")

    assert cleanup[0][:3] == ["docker", "rm", "--force"]


def test_requires_trusted_usage_directory_before_starting(tmp_path, monkeypatch):
    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')

    with pytest.raises(RunnerError, match="usage"):
        DockerRunner(IMAGE)(tmp_path, "task")

    assert commands == []


def test_reports_only_trusted_usage_and_measures_full_execution(tmp_path, monkeypatch):
    from evaluator.usage import add_usage

    fake_docker(
        monkeypatch,
        'import time; time.sleep(0.02); print(\'{"answer":"ok","usage":{"total_tokens":0}}\')',
    )
    exchange = DockerRunner._exchange

    def complete(self, process, task):
        run_id = json.loads(task)["model_base_url"].split("/runs/")[1].split("/")[0]
        add_usage(
            self.usage_dir,
            run_id,
            {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
        )
        add_usage(
            self.usage_dir,
            run_id,
            {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        )
        return exchange(self, process, task)

    monkeypatch.setattr(DockerRunner, "_exchange", complete)
    result = DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")

    assert isinstance(result, SolveResult)
    assert result.answer == "ok"
    assert result.usage == {
        "prompt_tokens": 6,
        "completion_tokens": 9,
        "total_tokens": 15,
        "attempts": 2,
        "unreported_attempts": 0,
    }
    assert result.wall_seconds >= 0.02


def test_zero_model_calls_are_measured_as_zero(tmp_path, monkeypatch):
    fake_docker(monkeypatch, 'print(\'{"answer":"static"}\')')
    result = DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")

    assert result.usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "attempts": 0,
        "unreported_attempts": 0,
    }


def test_missing_ledger_is_an_error(tmp_path, monkeypatch):
    fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')

    def missing(*args):
        raise FileNotFoundError("secret path")

    monkeypatch.setattr("evaluator.runner.seal_usage", missing)
    with pytest.raises(RunnerError, match="^runner failed$"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")


def test_operator_timeouts_are_passed_to_harness(tmp_path, monkeypatch):
    script = (
        "import json,sys; task=json.load(sys.stdin); print(json.dumps({'answer':json.dumps(task)}))"
    )
    fake_docker(monkeypatch, script)
    runner = DockerRunner(
        IMAGE,
        usage_dir=tmp_path / "usage",
        model_timeout=10,
        retry_delay=2,
        prepare_timeout=80,
    )
    task = json.loads(runner(tmp_path, "task").answer)

    assert task["model_request_timeout"] == 182
    assert task["prepare_timeout"] == 80
    assert runner.timeout == 604800


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_timeout", 0),
        ("model_timeout", True),
        ("model_timeout", float("inf")),
        ("prepare_timeout", -1),
        ("prepare_timeout", float("nan")),
        ("retry_delay", -1),
        ("retry_delay", False),
        ("retry_delay", float("inf")),
    ],
)
def test_rejects_invalid_operation_limits(field, value):
    with pytest.raises(ValueError, match="limits"):
        DockerRunner(IMAGE, **{field: value})


def test_example_harness_preparation_uses_operator_timeout(monkeypatch):
    timeouts = []
    inputs = []

    def prepare(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        inputs.append(json.loads(kwargs["input"]))
        kwargs["stdout"].write(b'{"context":"prepared"}')

    monkeypatch.setattr(sample_runner, "read_limited", lambda path: "candidate prompt")
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(sample_runner.subprocess, "run", prepare)
    monkeypatch.setattr(
        sample_runner.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(
            open=lambda request, timeout: io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')
        ),
    )

    assert (
        sample_runner.solve(
            {
                "input": "task",
                "model": "internal",
                "model_base_url": "http://gateway/runs/test/v1",
                "model_request_timeout": 130410,
                "prepare_timeout": 172800,
            }
        )
        == "ok"
    )
    assert timeouts == [172800]
    assert inputs == [
        {
            "input": "task",
            "assets_dir": "/assets",
            "model": "internal",
            "model_base_url": "http://gateway/runs/test/v1",
            "model_request_timeout": 130410,
        }
    ]


def test_example_harness_does_not_repeat_gateway_retries(monkeypatch):
    calls = []

    def fail(request, timeout):
        calls.append(request)
        raise TimeoutError

    monkeypatch.setattr(sample_runner, "read_limited", lambda path: "candidate prompt")
    monkeypatch.setattr(Path, "is_file", lambda path: False)
    monkeypatch.setattr(
        sample_runner.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=fail),
    )
    with pytest.raises(TimeoutError):
        sample_runner.solve(
            {
                "input": "task",
                "model": "internal",
                "model_base_url": "http://gateway/runs/test/v1",
                "model_request_timeout": 130410,
                "prepare_timeout": 172800,
            }
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://gateway",
        "http://gateway/custom/v1",
        "http://user:secret@gateway/v1",
        "http://gateway/v1?override=true",
        "http://gateway/v1#fragment",
        "file:///v1",
    ],
)
def test_requires_canonical_gateway_base_url(url):
    with pytest.raises(ValueError, match="gateway URL"):
        DockerRunner(IMAGE, model_base_url=url)


def test_unknown_model_usage_is_preserved(tmp_path, monkeypatch):
    from evaluator.usage import add_usage

    fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    exchange = DockerRunner._exchange

    def complete(self, process, task):
        run_id = json.loads(task)["model_base_url"].split("/runs/")[1].split("/")[0]
        add_usage(self.usage_dir, run_id, None)
        return exchange(self, process, task)

    monkeypatch.setattr(DockerRunner, "_exchange", complete)
    result = DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")

    assert result.usage["attempts"] == 1
    assert result.usage["unreported_attempts"] == 1


def test_pending_model_call_prevents_success(tmp_path, monkeypatch):
    from evaluator.usage import begin_request

    fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    exchange = DockerRunner._exchange

    def complete(self, process, task):
        run_id = json.loads(task)["model_base_url"].split("/runs/")[1].split("/")[0]
        begin_request(self.usage_dir, run_id)
        return exchange(self, process, task)

    monkeypatch.setattr(DockerRunner, "_exchange", complete)
    with pytest.raises(RunnerError, match="^runner failed$"):
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage")(tmp_path, "task")


@pytest.mark.parametrize("failed", [False, True])
def test_model_scope_is_closed_after_container_exit(tmp_path, monkeypatch, failed):
    from evaluator.usage import begin_request

    script = "import sys; sys.exit(1)" if failed else 'print(\'{"answer":"ok"}\')'
    commands, _ = fake_docker(monkeypatch, script)
    usage_dir = tmp_path / "usage"
    runner = DockerRunner(IMAGE, usage_dir=usage_dir)
    if failed:
        with pytest.raises(RunnerError, match="^runner failed$"):
            runner(tmp_path, "task")
    else:
        runner(tmp_path, "task")
    name = commands[0][commands[0].index("--name") + 1]
    run_id = name.removeprefix("arena-runner-")

    with pytest.raises(ValueError):
        begin_request(usage_dir, run_id)


def test_runner_error_exposes_optional_private_metrics():
    error = RunnerError("failure")

    assert error.usage is None
    assert error.wall_seconds is None
    assert error.run_id is None
    assert str(error) == "failure"


def test_failure_retains_charged_attempts_and_full_elapsed_time(tmp_path, monkeypatch):
    from evaluator.usage import add_usage, read_usage

    fake_docker(monkeypatch, "import time,sys; time.sleep(0.02); sys.exit(1)")
    exchange = DockerRunner._exchange
    run_ids = []

    def complete(self, process, task):
        run_id = json.loads(task)["model_base_url"].split("/runs/")[1].split("/")[0]
        run_ids.append(run_id)
        add_usage(
            self.usage_dir,
            run_id,
            {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
        )
        add_usage(self.usage_dir, run_id, None)
        return exchange(self, process, task)

    monkeypatch.setattr(DockerRunner, "_exchange", complete)
    usage_dir = tmp_path / "usage"
    with pytest.raises(RunnerError, match="^runner failed$") as failure:
        DockerRunner(IMAGE, usage_dir=usage_dir)(tmp_path, "private task")

    error = failure.value
    assert error.run_id == run_ids[0]
    assert error.wall_seconds is not None and error.wall_seconds >= 0.02
    assert error.usage == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
        "attempts": 2,
        "unreported_attempts": 1,
    }
    assert read_usage(usage_dir, run_ids[0]) == error.usage
