import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).parents[1] / "campaigns/sc35-first-20261028/deploy"
ROOT = Path(__file__).parents[3]


def compose_config(tmp_path, path, **overrides):
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("ARENA_", "SC35_", "SC71_", "COMPOSE_"))
    }
    environment.update({
        "ARENA_ROOT": str(tmp_path), "ARENA_DOCKER_GID": "1000",
        "ARENA_EVALUATOR_IMAGE": "evaluator:synthetic",
        "ARENA_HERMES_IMAGE": "hermes:synthetic",
        "ARENA_CONTENT_IMAGE": "content:synthetic",
        "ARENA_INFRA_IMAGE": "infra:synthetic",
        "SC35_INPUT_ASSETS_DIGEST": "4" * 64,
        "SC35_MAX_SOLUTIONS": "100", "SC35_FINALISTS": "10", "SC35_OUTPUT_COUNT": "3",
        "SC35_PRIVATE_RELEASE_ID": "sc35-first-20261028",
        "SC35_MLFLOW_IMAGE": "mlflow:synthetic", "SC71_MLFLOW_IMAGE": "mlflow:synthetic",
        **overrides,
    })
    result = subprocess.run(
        [docker, "compose", "-f", str(path), "--profile", "*",
         "config", "--no-env-resolution", "--format", "json"],
        capture_output=True, text=True, check=True, env=environment,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("probe_status", [0, 42])
def test_launch_requires_mlflow_payload_before_creating_model_access(tmp_path, probe_status):
    operator = tmp_path / "operator"
    (operator / "launch").mkdir(parents=True)
    (tmp_path / "release").mkdir()
    (operator / "layout-version").write_text("sc35-first-20261028-layout-v1")
    (operator / "READY").touch()
    (operator / "images.env").write_text(
        "ARENA_INFRA_IMAGE=synthetic-infra\nARENA_EVALUATOR_IMAGE=synthetic-evaluator\n"
        "SC35_MAX_SOLUTIONS=100\nSC35_FINALISTS=10\nSC35_OUTPUT_COUNT=3\n"
        "SC35_PRIVATE_RELEASE_ID=sc35-first-20261028\n"
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
root = Path(os.environ["ARENA_ROOT"])
with (root / "docker-calls.jsonl").open("a") as output:
    output.write(json.dumps(arguments) + "\\n")
if "ps" in arguments and "--format" in arguments:
    for service in ("train-evaluator", "public-evaluator", "private-evaluator",
                    "model-gateway", "researcher-gateway", "internet-proxy",
                    "research-monitor"):
        print(json.dumps({"Service": service, "State": "running", "Health": "healthy"}))
elif any(value.endswith("/mlflow_probe.py") for value in arguments):
    raise SystemExit(int(os.environ["TEST_PROBE_STATUS"]))
elif "-c" in arguments and "create_usage" in arguments[-1]:
    (root / "operator/launch/active.json").write_text('{"synthetic": true}')
    (root / "model-usage-created").touch()
elif "up" in arguments:
    (root / "search-started").touch()
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
    result = subprocess.run(
        ["bash", str(DEPLOY / "launch.sh")],
        env=dict(os.environ, ARENA_ROOT=str(tmp_path), TEST_PROBE_STATUS=str(probe_status),
                 PATH=str(binary) + os.pathsep + os.environ["PATH"]),
        capture_output=True, text=True, check=False,
    )
    calls = [json.loads(row) for row in (tmp_path / "docker-calls.jsonl").read_text().splitlines()]
    probes = [row for row in calls if any(value.endswith("/mlflow_probe.py") for value in row)]
    assert len(probes) == 1
    probe = probes[0]
    assert probe[probe.index("run"):probe.index("researcher")] == [
        "run", "--rm", "--no-deps", "-T", "--entrypoint", "/opt/arena/.venv/bin/python",
    ]
    assert probe[-3:] == ["read", "--receipt", "/run/arena/mlflow-probe-receipt.json"]
    assert result.returncode == probe_status, result.stderr
    assert (operator / "launch/active.json").exists() is (probe_status == 0)
    assert (tmp_path / "model-usage-created").exists() is (probe_status == 0)
    assert (tmp_path / "search-started").exists() is (probe_status == 0)
    capabilities = [row for row in calls if "create_usage" in str(row)]
    if probe_status == 0:
        assert calls.index(probe) < calls.index(capabilities[0])
    else:
        assert not capabilities


@pytest.mark.parametrize("active", [False, True])
def test_launch_refuses_unprepared_or_initialized_campaign_before_docker(tmp_path, active):
    operator = tmp_path / "operator"
    (operator / "launch").mkdir(parents=True)
    (operator / "READY").write_text("synthetic")
    (operator / "layout-version").write_text(
        "sc35-first-20261028-layout-v1" if active else "sc71-second-20261024-layout-v2"
    )
    if active:
        (operator / "launch/active.json").write_text('{"run_id":"existing"}')
    binary = tmp_path / "docker"
    binary.write_text('#!/bin/sh\ntouch "$ARENA_ROOT/docker-called"\nexit 97\n')
    binary.chmod(0o755)
    result = subprocess.run(
        ["bash", str(DEPLOY / "launch.sh")],
        env=dict(os.environ, ARENA_ROOT=str(tmp_path),
                 PATH=str(tmp_path) + os.pathsep + os.environ["PATH"]),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert not (tmp_path / "docker-called").exists()
    assert "already initialized" in result.stderr if active else "readiness" in result.stderr


def test_firewall_preserves_existing_campaigns(tmp_path):
    binary = tmp_path / "iptables-legacy"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_CALLS"\n')
    binary.chmod(0o755)
    log = tmp_path / "calls"
    subprocess.run(
        ["bash", str(DEPLOY / "firewall.sh")], check=True,
        env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                 TEST_CALLS=str(log)),
    )
    calls = log.read_text().splitlines()
    assert {row for row in calls if row.startswith("-F ")} == {
        "-F ARENA_SC35_1_FORWARD", "-F ARENA_SC35_1_INPUT",
    }
    assert "-A ARENA_SC35_1_FORWARD -s 172.31.91.0/24 -j REJECT" in calls
    assert "-A ARENA_SC35_1_FORWARD -s 172.31.92.0/24 -j REJECT" in calls
    assert not any("SC71" in row for row in calls)
    config = compose_config(tmp_path, DEPLOY / "compose.yaml")
    sources = set()
    for service in ("train-evaluator", "public-evaluator", "private-evaluator"):
        addresses = config["services"][service]["networks"]
        assert all(network.get("ipv4_address") for network in addresses.values())
        sources.update(network["ipv4_address"] for network in addresses.values())
    targets = {"172.31.91.20", "172.31.93.20", "172.31.95.2", "172.31.95.3", "172.28.0.2"}
    blocked = []
    for index, call in enumerate(calls):
        words = shlex.split(call)
        if "--dport" in words and words[words.index("-j") + 1] == "REJECT":
            assert words[words.index("-p") + 1] == "tcp"
            assert words[words.index("--dport") + 1] == "5000"
            blocked.append((words[words.index("-s") + 1], words[words.index("-d") + 1]))
            assert index < calls.index(
                "-A ARENA_SC35_1_FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT"
            )
    assert set(blocked) == {(source, target) for source in sources for target in targets}
    assert len(blocked) == len(sources) * len(targets)


def test_compose_isolates_sc35_and_keeps_search_opt_in(tmp_path):
    config = compose_config(tmp_path, DEPLOY / "compose.yaml")
    assert all(item["name"].startswith("sc35-first-") for item in config["networks"].values())
    researcher = config["services"]["researcher"]
    assert researcher["profiles"] == ["search"]
    assert set(researcher["networks"]) == {"research"}
    assert all("docker.sock" not in item["target"] for item in researcher["volumes"])
    assert all("operator" not in item.get("source", "") for item in researcher["volumes"])
    private = config["services"]["private-evaluator"]
    settings = private["environment"]
    assert settings["ARENA_SEARCH_MODE"] == "campaign"
    assert settings["ARENA_SEARCH_MAX_SOLUTIONS"] == "100"
    assert settings["ARENA_FINALISTS"] == "10"
    assert settings["ARENA_OUTPUT_COUNT"] == "3"
    assert settings["ARENA_RELEASE_ID"] == "sc35-first-20261028"
    mounts = {item["target"]: item for item in private["volumes"]}
    assert mounts["/search"]["read_only"] is True
    assert mounts["/operator-search"]["read_only"] is True
    harnesses = json.loads(settings["ARENA_HARNESSES"])
    assert harnesses["hermes"]["supports_build"] is True
    assert harnesses["sc35-content"]["supports_build"] is False
    gateway = config["services"]["model-gateway"]["environment"]
    assert gateway["MODEL_STRICT_SCHEMA_INTEGERS"] == "false"


@pytest.mark.parametrize("override", [None, "monitor:updated"])
def test_only_monitor_delivers_tracking_and_can_update_independently(tmp_path, override):
    overrides = {"ARENA_MONITOR_IMAGE": override} if override is not None else {}
    config = compose_config(tmp_path, DEPLOY / "compose.yaml", **overrides)
    for name in ("train-evaluator", "public-evaluator", "private-evaluator"):
        service = config["services"][name]
        assert service["environment"]["ARENA_TRACKING_DELIVERY"] == "queued"
        assert service["image"] == "evaluator:synthetic"
    monitor = config["services"]["research-monitor"]
    assert monitor["environment"].get("ARENA_TRACKING_DELIVERY", "inline") == "inline"
    assert monitor["environment"]["TMPDIR"] == str(
        tmp_path / "operator/state/research-monitor/tmp"
    )
    assert monitor["image"] == (override or "evaluator:synthetic")


@pytest.mark.parametrize("relative", [
    "cases/sc35/campaigns/sc35-first-20261028/deploy/mlflow-compose.yaml",
    "cases/sc71/campaigns/sc71-second-20261024/deploy/mlflow-compose.yaml",
    "cases/sc71/deploy/gpu6/mlflow/compose.yaml",
])
def test_shared_tracking_uses_one_worker_with_existing_memory_limit(tmp_path, relative):
    config = compose_config(tmp_path, ROOT / relative)
    service = config["services"]["shared-tracking"]
    command = service["command"]
    assert command[command.index("--workers") + 1] == "1"
    assert int(service["mem_limit"]) == 4 * 1024**3
