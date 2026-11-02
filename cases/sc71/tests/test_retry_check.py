import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "campaigns/sc71-second-20261024/deploy/retry_check.py"
SPEC = importlib.util.spec_from_file_location("sc71_retry_check", PATH)
assert SPEC is not None and SPEC.loader is not None
retry_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry_check)

NEW_DELAYS = [30, 90, 240, 540, 900]
OLD_DELAYS = [30, 60, 120, 240, 300]
SERVICES = (
    "model-gateway", "researcher-gateway", "train-evaluator", "public-evaluator",
    "private-evaluator", "researcher",
)


class Docker:
    def __init__(self, old=(), researcher=True, oneoff=False):
        self.old = set(old)
        self.oneoff = oneoff
        self.services = SERVICES if researcher else SERVICES[:-1]
        self.calls = []
        self.images = {name: "sha256:" + f"{index:064x}" for index, name in enumerate(
            (*SERVICES, "solver", "alternate"), 1,
        )}
        self.containers = {name: f"{index:064x}" for index, name in enumerate(SERVICES, 100)}

    def timing(self, role, hermes=False):
        old = role in self.old
        data = {
            "retry_delays": OLD_DELAYS if old else NEW_DELAYS,
            "request_timeout": 130410 if old else 131460,
        }
        if hermes:
            data.update(
                hermes_timeout=data["request_timeout"], config_timeout=data["request_timeout"],
            )
        return data

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        assert command[0] == "docker"
        assert kwargs["capture_output"] and kwargs["text"]
        args = command[1:]
        if args[0] == "ps":
            output = "\n".join(self.containers[name] for name in self.services)
        elif args[0] == "inspect":
            rows = [
                {
                    "Id": self.containers[name], "Image": self.images[name],
                    "State": {"Status": "exited" if name == "researcher" else "running"},
                    "Config": {
                        "Labels": {"com.docker.compose.service": name},
                        "Env": ["MODEL_API_KEY=do-not-leak", "MODEL_BASE_URL=http://private-secret"],
                    },
                }
                for name in self.services
            ]
            if self.oneoff:
                extra = {**rows[-2], "Config": {"Labels": {
                    "com.docker.compose.service": "private-evaluator",
                    "com.docker.compose.oneoff": "True",
                }}}
                rows.append(extra)
            output = json.dumps(rows)
        elif args[:2] == ["image", "inspect"]:
            image = args[2]
            name = image if image in self.images else next(
                name for name, identifier in self.images.items() if identifier == image
            )
            output = json.dumps([{"Id": self.images[name]}])
        elif args[0] == "exec":
            name = next(name for name, identifier in self.containers.items() if identifier in args)
            data = self.timing(name)
            if name.endswith("evaluator"):
                data.update(runner_image="solver", harnesses={"hermes": "alternate"})
            output = json.dumps(data)
        elif args[0] == "run":
            assert "--network" in args and args[args.index("--network") + 1] == "none"
            assert "--read-only" in args
            name = next(name for name, identifier in self.images.items() if identifier in args)
            mode = "credentials" if args[-1] == "researcher-access" else name
            output = json.dumps(self.timing(mode, hermes=mode != "credentials"))
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="secret-from-stderr")


def test_mixed_rollout_rejects_old_private_even_when_researcher_is_updated(monkeypatch, capsys):
    docker = Docker(old={"private-evaluator"})
    monkeypatch.setattr(retry_check.subprocess, "run", docker)
    assert retry_check.main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert any(item["role"] == "private-evaluator" for item in report["failures"])
    assert any(item["role"] == "researcher" and item["request_timeout"] == 131460
               for item in report["checks"])


def test_complete_rollout_checks_every_runtime_and_does_not_print_secrets(monkeypatch, capsys):
    docker = Docker()
    monkeypatch.setattr(retry_check.subprocess, "run", docker)
    assert retry_check.main([]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["status"] == "ok" and report["failures"] == []
    roles = {row["role"] for row in report["checks"]}
    assert set(SERVICES) <= roles
    for service in ("train-evaluator", "public-evaluator", "private-evaluator"):
        assert f"{service}/runner" in roles
        assert f"{service}/harness/hermes" in roles
    assert all(row["retry_delays"] == NEW_DELAYS for row in report["checks"])
    assert all(row["request_timeout"] == 131460 for row in report["checks"])
    assert "do-not-leak" not in output and "private-secret" not in output
    assert "secret-from-stderr" not in output and "MODEL_API_KEY" not in output
    assert all("Config" not in row for row in report["checks"])


@pytest.mark.parametrize("old", [
    "model-gateway", "researcher-gateway", "solver", "alternate", "researcher",
])
def test_each_model_path_can_block_rollout(monkeypatch, old):
    monkeypatch.setattr(retry_check.subprocess, "run", Docker(old={old}))
    report = retry_check.audit("arena-sc71-second")
    assert report["status"] == "failed" and report["failures"]


def test_initial_launch_requires_explicit_researcher_image(monkeypatch):
    monkeypatch.setattr(retry_check.subprocess, "run", Docker(researcher=False))
    missing = retry_check.audit("arena-sc71-second")
    assert any(item["role"] == "researcher" for item in missing["failures"])
    ready = retry_check.audit("arena-sc71-second", researcher_image="researcher")
    assert ready["status"] == "ok"


def test_probe_error_hides_docker_stderr_and_arguments(monkeypatch, capsys):
    def fail(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="secret-output", stderr="secret-error",
        )

    monkeypatch.setattr(retry_check.subprocess, "run", fail)
    assert retry_check.main([]) == 1
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "failed"
    assert "secret-output" not in output and "secret-error" not in output


@pytest.mark.parametrize("mode", ["gateway", "evaluator", "hermes"])
def test_probe_reads_runtime_values_without_returning_credentials(mode):
    env = dict(os.environ, MODEL_API_KEY="private-key", MODEL_BASE_URL="http://secret-url",
               MODEL_RETRY_DELAY="30", MODEL_TIMEOUT="21600", ARENA_MODEL_TIMEOUT="21600",
               ARENA_RETRY_DELAY="30", ARENA_RUNNER_IMAGE="solver",
               ARENA_HARNESSES=json.dumps({"hermes": {"image": "alternate", "api_key": "hidden"}}))
    result = subprocess.run(
        [sys.executable, "-c", retry_check.PROBE, mode], capture_output=True, text=True,
        env=env, check=True,
    )
    data = json.loads(result.stdout)
    assert data["retry_delays"] == NEW_DELAYS and data["request_timeout"] == 131460
    if mode == "evaluator":
        assert data["runner_image"] == "solver" and data["harnesses"] == {"hermes": "alternate"}
    if mode == "hermes":
        assert data["hermes_timeout"] == data["config_timeout"] == 131460
    assert "private-key" not in result.stdout and "secret-url" not in result.stdout
    assert "hidden" not in result.stdout


def test_existing_researcher_credentials_cannot_override_new_image_with_old_timeout(monkeypatch):
    monkeypatch.setattr(retry_check.subprocess, "run", Docker(old={"credentials"}))
    report = retry_check.audit("arena-sc71-second")
    assert report["status"] == "failed"
    assert any(row["role"] == "researcher/access" for row in report["failures"])


def test_oneoff_compose_container_does_not_replace_or_duplicate_service(monkeypatch):
    monkeypatch.setattr(retry_check.subprocess, "run", Docker(oneoff=True))
    assert retry_check.audit("arena-sc71-second")["status"] == "ok"


def test_evaluator_probe_honors_file_configuration_before_environment(tmp_path):
    timeout_file = tmp_path / "timeout"
    timeout_file.write_text("60")
    image_file = tmp_path / "image"
    image_file.write_text("actual-solver")
    harness_file = tmp_path / "harnesses"
    harness_file.write_text(json.dumps({"hermes": {"image": "actual-alternate"}}))
    env = dict(os.environ, ARENA_MODEL_TIMEOUT="21600", ARENA_RETRY_DELAY="30",
               ARENA_MODEL_TIMEOUT_FILE=str(timeout_file), ARENA_RUNNER_IMAGE="ignored-solver",
               ARENA_RUNNER_IMAGE_FILE=str(image_file), ARENA_HARNESSES="{}",
               ARENA_HARNESSES_FILE=str(harness_file))
    result = subprocess.run(
        [sys.executable, "-c", retry_check.PROBE, "evaluator"], capture_output=True, text=True,
        env=env, check=True,
    )
    data = json.loads(result.stdout)
    assert data["request_timeout"] == 2220
    assert data["runner_image"] == "actual-solver"
    assert data["harnesses"] == {"hermes": "actual-alternate"}


def test_researcher_probe_reads_only_numeric_timeout_from_access_file(tmp_path):
    access = tmp_path / "model-access.json"
    access.write_text(json.dumps({"model_request_timeout": 130410,
                                  "model_base_url": "http://private-secret", "api_key": "hidden"}))
    result = subprocess.run(
        [sys.executable, "-c", retry_check.PROBE, "researcher-access"],
        capture_output=True, text=True,
        env=dict(os.environ, ARENA_MODEL_ACCESS_FILE=str(access)), check=True,
    )
    assert json.loads(result.stdout)["request_timeout"] == 130410
    assert "private-secret" not in result.stdout and "hidden" not in result.stdout


def test_duplicate_regular_service_still_fails_inventory(monkeypatch):
    docker = Docker()

    def duplicate(command, **kwargs):
        result = docker(command, **kwargs)
        if command[1] == "inspect":
            rows = json.loads(result.stdout)
            rows.append(rows[0])
            result.stdout = json.dumps(rows)
        return result

    monkeypatch.setattr(retry_check.subprocess, "run", duplicate)
    report = retry_check.audit("arena-sc71-second")
    assert report["status"] == "failed"
    assert any(row["role"] == "deployment" for row in report["failures"])
