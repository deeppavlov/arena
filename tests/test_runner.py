import hashlib
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


@pytest.mark.parametrize("image", [
    "python", "python:3.12", "image:latest", "registry.example:5000/team/runner:v1",
    IMAGE, "registry.example/team/runner@" + IMAGE,
])
def test_accepts_operator_image_references(image):
    assert DockerRunner(image).image == image


@pytest.mark.parametrize("image", [
    "", "--help", "image --privileged", "image\n--privileged", " image", "image ",
    "/image", "./image", "../image", "registry/../image", "registry//image",
    "image;command", "$(command)", "image\\other",
])
def test_rejects_unsafe_operator_image_references(image):
    with pytest.raises(ValueError, match="image"):
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
    run_id, capability, version = scoped_url.split("/runs/")[1].split("/")
    assert str(UUID(run_id)) == run_id
    assert version == "v1"
    assert len(capability) == 64
    assert all(character in "0123456789abcdef" for character in capability)
    assert run_id.replace("-", "") not in capability
    assert capability not in json.dumps(commands)
    assert capability not in json.dumps(runner.configuration())

    assert result == {
        "input": "private task",
        "model": "internal-model",
        "model_request_timeout": 131460,
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


@pytest.mark.parametrize("tmpfs_exec,mode", [(False, "noexec"), (True, "exec")])
def test_operator_resources_control_scratch_without_weakening_isolation(
    tmp_path, monkeypatch, tmpfs_exec, mode,
):
    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    runner = DockerRunner(
        IMAGE, uses_model=False, memory="32g", cpus="8", pids_limit=2048,
        tmpfs_size="16g", tmpfs_exec=tmpfs_exec,
    )

    runner(tmp_path, "task", parameters={"memory": "999g", "tmpfs_exec": not tmpfs_exec})

    command = commands[0]
    assert command[command.index("--memory") + 1] == "32g"
    assert command[command.index("--cpus") + 1] == "8"
    assert command[command.index("--pids-limit") + 1] == "2048"
    assert command[command.index("--tmpfs") + 1] == (
        f"/tmp:rw,{mode},nosuid,nodev,size=16g,uid=10001,gid=10001"
    )
    assert "--read-only" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--privileged" not in command


@pytest.mark.parametrize("field,value", [
    ("memory", "0g"), ("memory", "32g,exec"), ("memory", 32),
    ("cpus", "0"), ("cpus", "nan"), ("cpus", "9" * 400), ("cpus", 8),
    ("pids_limit", 0), ("pids_limit", True), ("pids_limit", 2.5),
    ("tmpfs_size", "0g"), ("tmpfs_size", "16g,suid"), ("tmpfs_size", 16),
    ("tmpfs_exec", "false"), ("tmpfs_exec", 1),
])
def test_resource_limits_reject_invalid_operator_configuration(field, value):
    with pytest.raises(ValueError, match="runner.*limit"):
        DockerRunner(IMAGE, **{field: value})


@pytest.mark.parametrize("uses_model", [False, True])
def test_success_returns_trusted_run_id_for_trace_correlation(tmp_path, monkeypatch, uses_model):
    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok","run_id":"forged"}\')')
    result = DockerRunner(
        IMAGE, uses_model=uses_model, usage_dir=tmp_path / "usage",
    )(tmp_path, "task")

    command = commands[0]
    name = command[command.index("--name") + 1]
    assert result.run_id == name.removeprefix("arena-runner-")
    assert result.run_id == str(UUID(result.run_id))
    assert SolveResult("ok", {}, 1).run_id is None


def test_resource_limits_are_frozen_in_harness_provenance():
    from evaluator.runner import HarnessRunner

    baseline = HarnessRunner(DockerRunner(IMAGE), {}).provenance()
    changed = HarnessRunner(
        DockerRunner(IMAGE, tmpfs_size="16g", tmpfs_exec=True), {},
    ).provenance()

    assert baseline["default"]["tmpfs_size"] == "128m"
    assert baseline["default"]["tmpfs_exec"] is False
    assert changed["default"]["tmpfs_size"] == "16g"
    assert changed["default"]["tmpfs_exec"] is True
    assert baseline["digest"] != changed["digest"]


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
        assert timeout == 131460
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
                "model_request_timeout": 131460,
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

    assert task["model_request_timeout"] == 240
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
                "model_request_timeout": 131460,
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
            "model_request_timeout": 131460,
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
                "model_request_timeout": 131460,
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
        capability = json.loads(task)["model_base_url"].split("/runs/")[1].split("/")[1]
        begin_request(self.usage_dir, run_id, capability)
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
    credentials = []
    exchange = DockerRunner._exchange

    def capture(self, process, task):
        credentials.append(json.loads(task)["model_base_url"].split("/runs/")[1].split("/")[1])
        return exchange(self, process, task)

    monkeypatch.setattr(DockerRunner, "_exchange", capture)
    runner = DockerRunner(IMAGE, usage_dir=usage_dir)
    if failed:
        with pytest.raises(RunnerError, match="^runner failed$"):
            runner(tmp_path, "task")
    else:
        runner(tmp_path, "task")
    name = commands[0][commands[0].index("--name") + 1]
    run_id = name.removeprefix("arena-runner-")

    with pytest.raises(ValueError, match="closed"):
        begin_request(usage_dir, run_id, credentials[0])


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


def test_model_free_runner_needs_no_model_or_ledger(tmp_path, monkeypatch):
    commands, _ = fake_docker(
        monkeypatch,
        "import json,sys,time; task=json.load(sys.stdin); time.sleep(.02); "
        "print(json.dumps({'answer':json.dumps(task),'usage':{'total_tokens':999}}))",
    )
    result = DockerRunner(IMAGE, uses_model=False)(tmp_path, "task")

    assert json.loads(result.answer) == {
        "input": "task", "uses_model": False, "prepare_timeout": 172800,
    }
    assert result.usage == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "attempts": 0, "unreported_attempts": 0,
    }
    assert result.wall_seconds >= .02
    assert commands[0][commands[0].index("--network") + 1] == "none"


def test_model_free_runner_rejects_network_access():
    with pytest.raises(ValueError, match="model-free.*network"):
        DockerRunner(IMAGE, uses_model=False, network="workers")


def test_model_free_failure_retains_zero_usage_and_time(tmp_path, monkeypatch):
    fake_docker(monkeypatch, "import time,sys; time.sleep(.02); sys.exit(1)")

    with pytest.raises(RunnerError) as failure:
        DockerRunner(IMAGE, uses_model=False)(tmp_path, "task")

    assert failure.value.usage is not None
    assert failure.value.usage["total_tokens"] == 0
    assert failure.value.usage["unreported_attempts"] == 0
    assert failure.value.wall_seconds is not None and failure.value.wall_seconds >= .02


def test_harness_selection_uses_registered_profile_and_nested_parameters(tmp_path, monkeypatch):
    from evaluator.runner import HarnessRunner

    second_image = "sha256:" + "b" * 64
    manifest = {
        "harness": "program",
        "parameters": {"depth": 4, "model": "fake", "image": "untrusted"},
    }
    (tmp_path / "solution.json").write_text(json.dumps(manifest))
    commands, _ = fake_docker(
        monkeypatch,
        "import json,sys; task=json.load(sys.stdin); "
        "print(json.dumps({'answer':json.dumps(task)}))",
    )
    runner = HarnessRunner(
        DockerRunner(IMAGE, usage_dir=tmp_path / "usage"),
        {"program": DockerRunner(second_image, uses_model=False)},
    )
    result = json.loads(runner(tmp_path, "task").answer)

    assert commands[0][-1] == second_image
    assert result["parameters"] == manifest["parameters"]
    assert "model" not in result
    assert "image" not in result
    assert "model_base_url" not in result


@pytest.mark.parametrize("manifest", [
    {"harness": "unknown"},
    {"harness": "program", "image": IMAGE},
    {"harness": "program", "network": "workers"},
    {"harness": "program", "model_base_url": "http://host/v1"},
    {"harness": "program", "parameters": []},
    {"harness": "program", "parameters": {"depth": float("nan")}},
    {"harness": "program", "parameters": {"depth": float("inf")}},
    {"parameters": {}},
    [],
])
def test_invalid_manifest_fails_before_starting(tmp_path, monkeypatch, manifest):
    from evaluator.runner import HarnessRunner

    (tmp_path / "solution.json").write_text(json.dumps(manifest))
    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    runner = HarnessRunner(
        DockerRunner(IMAGE), {"program": DockerRunner(IMAGE, uses_model=False)}
    )

    with pytest.raises(RunnerError, match="harness"):
        runner(tmp_path, "task")

    assert commands == []


def test_harness_registry_falls_back_to_default_without_manifest(tmp_path, monkeypatch):
    from evaluator.runner import HarnessRunner

    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    runner = HarnessRunner(DockerRunner(IMAGE, uses_model=False), {})

    assert runner(tmp_path, "task").answer == "ok"
    assert commands[0][-1] == IMAGE


def test_harness_provenance_freezes_configuration_and_order():
    from evaluator.runner import HarnessRunner

    default = DockerRunner(IMAGE, uses_model=False)
    registry = {"b": default, "a": DockerRunner(IMAGE, timeout=100)}
    runner = HarnessRunner(default, registry)
    first = runner.provenance()
    registry.clear()

    assert runner.provenance() == first
    assert first["default"]["uses_model"] is False
    assert first["registry"]["a"]["timeout"] == 100
    assert first["digest"] == HarnessRunner(default, {
        "a": DockerRunner(IMAGE, timeout=100), "b": default,
    }).provenance()["digest"]
    assert first["digest"] != HarnessRunner(default, {
        "a": DockerRunner(IMAGE, timeout=101), "b": default,
    }).provenance()["digest"]


def test_invalid_derived_request_timeout_does_not_leave_open_scope(tmp_path, monkeypatch):
    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    usage_dir = tmp_path / "usage"
    runner = DockerRunner(IMAGE, model_timeout=1e308, usage_dir=usage_dir)

    with pytest.raises(ValueError, match="request timeout"):
        runner(tmp_path, "task")

    assert commands == []
    assert not usage_dir.exists()


@pytest.mark.parametrize("uses_model", [False, True])
@pytest.mark.parametrize("character", ["я", "😀", "\x00"])
def test_maximum_example_and_manifest_reach_sample_runner(
    tmp_path, monkeypatch, uses_model, character,
):
    from evaluator.contracts import Example
    from evaluator.runner import HarnessRunner

    example = Example(id="maximum", input=character * 500000)
    prefix = '{"harness":"sample","parameters":{"values":['
    suffix = ']}}'
    count = (1024 * 1024 - len(prefix) - len(suffix) + 1) // 5
    manifest = prefix + ",".join(["1e15"] * count) + suffix
    assert 1024 * 1024 - 5 < len(manifest) <= 1024 * 1024
    (tmp_path / "solution.json").write_text(manifest)
    script = (
        "import hashlib,json; from runner import main; "
        "main.solve = lambda task: hashlib.sha256(json.dumps("
        "[task['input'],task['parameters']],ensure_ascii=False).encode()).hexdigest(); "
        "raise SystemExit(main.main())"
    )
    fake_docker(monkeypatch, script)
    profile = DockerRunner(
        IMAGE, uses_model=uses_model, model="internal", usage_dir=tmp_path / "usage",
    )
    runner = HarnessRunner(profile, {"sample": profile})

    result = runner(tmp_path, example.input)

    expected = hashlib.sha256(json.dumps(
        [example.input, json.loads(manifest)["parameters"]], ensure_ascii=False,
    ).encode()).hexdigest()
    assert result.answer == expected


@pytest.mark.parametrize("uses_model", [False, True])
def test_oversized_task_fails_before_starting_container(tmp_path, monkeypatch, uses_model):
    from evaluator.usage import ensure_open

    commands, _ = fake_docker(monkeypatch, 'print(\'{"answer":"ok"}\')')
    runner = DockerRunner(
        IMAGE, uses_model=uses_model, model="internal", usage_dir=tmp_path / "usage",
    )

    with pytest.raises(RunnerError, match="task too large") as failure:
        runner(tmp_path, "x" * (8 * 1024 * 1024))

    assert commands == []
    if uses_model:
        assert failure.value.run_id is not None
        with pytest.raises(ValueError, match="closed"):
            ensure_open(tmp_path / "usage", failure.value.run_id)


@pytest.mark.parametrize("uses_model", [False, True])
def test_task_uses_utf8_and_preserves_surrogate_parameters(tmp_path, monkeypatch, uses_model):
    from evaluator.runner import HarnessRunner

    label = "я😀\ud800"
    (tmp_path / "solution.json").write_text(json.dumps({
        "harness": "sample", "parameters": {"label": label},
    }))
    fake_docker(monkeypatch, (
        "from runner import main; main.solve = lambda task: task['parameters']['label']; "
        "raise SystemExit(main.main())"
    ))
    exchange = DockerRunner._exchange
    payloads = []

    def capture(self, process, task):
        payloads.append(task)
        return exchange(self, process, task)

    monkeypatch.setattr(DockerRunner, "_exchange", capture)
    profile = DockerRunner(
        IMAGE, uses_model=uses_model, model="internal", usage_dir=tmp_path / "usage",
    )

    result = HarnessRunner(profile, {"sample": profile})(tmp_path, "я😀")

    assert result.answer == label
    assert "я😀".encode() in payloads[0]
