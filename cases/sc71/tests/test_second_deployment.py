import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).parents[1] / "campaigns/sc71-second-20261024/deploy"
SERVICES = ("train-evaluator", "public-evaluator", "private-evaluator")


@pytest.fixture
def launch_environment(tmp_path):
    operator = tmp_path / "operator"
    (operator / "launch").mkdir(parents=True)
    (tmp_path / "release").mkdir()
    (operator / "layout-version").write_text("sc71-second-20261024-layout-v2")
    (operator / "READY").touch()
    (operator / "images.env").write_text(
        "ARENA_INFRA_IMAGE=synthetic-infra\nARENA_EVALUATOR_IMAGE=synthetic-evaluator\n"
    )
    for service in SERVICES:
        (operator / service).mkdir(mode=0o700)
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''import json
import os
import subprocess
import sys
from pathlib import Path

arguments = sys.argv[1:]
root = Path(os.environ["ARENA_ROOT"])
with (root / "docker-calls.jsonl").open("a") as output:
    output.write(json.dumps(arguments) + "\\n")
if "exec" in arguments:
    index = arguments.index("exec")
    assert arguments[index + 1] == "-T"
    service = arguments[index + 2]
    assert arguments[index + 3:index + 5] == ["python", "-c"]
    environment = dict(os.environ, ARENA_SLOT_DIR=str(root / "operator" / service),
                       ARENA_EXECUTION_SLOTS="1")
    result = subprocess.run([sys.executable, "-c", arguments[index + 5]], env=environment)
    raise SystemExit(result.returncode)
if "ps" in arguments and "--format" in arguments:
    for service in ("train-evaluator", "public-evaluator", "private-evaluator",
                    "model-gateway", "researcher-gateway", "internet-proxy",
                    "research-monitor"):
        print(json.dumps({"Service": service, "State": "running", "Health": "healthy"}))
elif "-c" in arguments and "create_usage" in arguments[-1]:
    (root / "operator/launch/active.json").write_text('{"synthetic": true}')
elif "-aq" in arguments or arguments[0] == "inspect":
    print("synthetic-container")
''')
    docker.chmod(0o755)
    cat = binary / "cat"
    cat.write_text('''#!/bin/sh
if [ "$1" = /proc/sys/net/bridge/bridge-nf-call-iptables ]; then
  printf '1\\n'
else
  exec /bin/cat "$@"
fi
''')
    cat.chmod(0o755)
    return dict(os.environ, ARENA_ROOT=str(tmp_path),
                PATH=str(binary) + os.pathsep + os.environ["PATH"],
                PYTHONPATH=str(Path(__file__).resolve().parents[3]))


@pytest.mark.parametrize("service", SERVICES)
@pytest.mark.parametrize("failure", ["missing", "writable-by-group"])
def test_second_launch_rejects_unusable_capacity_before_model_access(
    tmp_path, launch_environment, service, failure,
):
    directory = tmp_path / "operator" / service
    if failure == "missing":
        directory.rmdir()
    else:
        directory.chmod(0o770)
    result = subprocess.run(
        ["bash", str(DEPLOY / "launch.sh")], env=launch_environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "operator/launch/active.json").exists()
    calls = [json.loads(row) for row in (tmp_path / "docker-calls.jsonl").read_text().splitlines()]
    assert not any("create_usage" in str(row) for row in calls)
    assert not any("up" in row for row in calls)
    assert directory.exists() is (failure != "missing")


def test_second_launch_releases_all_capacity_probes_before_model_access(
    tmp_path, launch_environment,
):
    from evaluator.capacity import acquire_slot

    result = subprocess.run(
        ["bash", str(DEPLOY / "launch.sh")], env=launch_environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(row) for row in (tmp_path / "docker-calls.jsonl").read_text().splitlines()]
    capability_index = next(index for index, row in enumerate(calls) if "create_usage" in str(row))
    checked = [row[row.index("exec") + 2] for row in calls[:capability_index] if "exec" in row]
    assert checked == list(SERVICES)
    for service in SERVICES:
        directory = tmp_path / "operator" / service
        assert (directory / "slot-0.lock").exists()
        lease = acquire_slot(directory, 1)
        assert lease is not None
        lease.close()


def test_second_campaign_compose_has_independent_state(tmp_path):
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required")
    values = {
        "ARENA_ROOT": str(tmp_path), "ARENA_DOCKER_GID": "1000",
        "ARENA_EVALUATOR_IMAGE": "evaluator:synthetic",
        "ARENA_HERMES_IMAGE": "hermes:synthetic",
        "ARENA_INFRA_IMAGE": "infra:synthetic",
        "SC71_INPUT_ASSETS_DIGEST": "4" * 64,
        "SC71_FINALISTS": "10", "SC71_OUTPUT_COUNT": "3",
        "SC71_PRIVATE_RELEASE_ID": "sc71-second-20261024",
    }
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("ARENA_", "SC71_", "COMPOSE_"))
    }
    environment.update(values)
    result = subprocess.run(
        [docker, "compose", "-f", str(DEPLOY / "compose.yaml"), "--profile", "*",
         "config", "--no-env-resolution", "--format", "json"],
        capture_output=True, text=True, check=True, env=environment,
    )
    config = json.loads(result.stdout)
    assert config["name"] == "arena-sc71-second"
    assert all(row["name"].startswith("sc71-second-") for row in config["networks"].values())
    assert config["volumes"]["campaign-runtime"]["name"] == "sc71-second-campaign-runtime"
    assert "sc71-first-20261019" not in result.stdout
    services = config["services"]
    for service, port in (("train-evaluator", "18830"),
                          ("public-evaluator", "18831"),
                          ("private-evaluator", "18832")):
        assert services[service]["ports"][0]["published"] == port
        assert services[service]["ports"][0]["host_ip"] == "127.0.0.1"
    private = services["private-evaluator"]
    assert private["environment"]["ARENA_SEARCH_MODE"] == "campaign"
    assert private["environment"]["ARENA_SEARCH_MAX_SOLUTIONS"] == "100"
    assert private["environment"]["ARENA_RELEASE_ID"] == "sc71-second-20261024"
    volumes = {row["target"]: row for row in private["volumes"]}
    assert volumes["/search"]["read_only"] is True
    assert volumes["/operator-search"]["read_only"] is True
    researcher = services["researcher"]
    assert researcher["profiles"] == ["search"]
    assert researcher["environment"]["ARENA_CAMPAIGN"] == "sc71-second-20261024"
    assert all("docker.sock" not in row["target"] for row in researcher["volumes"])
    assert set(researcher["networks"]) == {"research"}


def test_second_launch_requires_its_own_ready_marker(tmp_path):
    assert (DEPLOY / "launch.sh").is_file()
    operator = tmp_path / "operator"
    operator.mkdir()
    (operator / "READY").write_text("historical readiness")
    (operator / "layout-version").write_text("sc71-case-layout-v2")
    result = subprocess.run(
        ["bash", str(DEPLOY / "launch.sh")],
        env=dict(os.environ, ARENA_ROOT=str(tmp_path)),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "sc71-second-20261024" in result.stderr
    assert not (operator / "launch").exists()


def test_second_firewall_only_changes_its_own_chains(tmp_path):
    log = tmp_path / "calls"
    binary = tmp_path / "iptables-legacy"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FIREWALL_TEST_LOG"\n')
    binary.chmod(0o755)
    subprocess.run(
        ["bash", str(DEPLOY / "firewall.sh")], check=True,
        env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                 FIREWALL_TEST_LOG=str(log)),
    )
    calls = log.read_text().splitlines()
    assert set(row for row in calls if row.startswith("-F ")) == {
        "-F ARENA_SC71_2_FORWARD", "-F ARENA_SC71_2_INPUT",
    }
    assert "-A ARENA_SC71_2_FORWARD -s 172.31.82.0/24 -j REJECT" in calls
    assert "-A ARENA_SC71_2_FORWARD -s 172.31.81.0/24 -j REJECT" in calls
    assert not any("172.31.71." in row or "172.31.72." in row for row in calls)
