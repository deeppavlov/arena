import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

PREFIX = "sc35-http"


def docker(*arguments, check=True):
    result = subprocess.run(
        ["docker", *map(str, arguments)], capture_output=True, text=True, timeout=360,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def mount(source, target=None, readonly=False):
    return ["--mount", "type=bind,src=" + str(source) + ",dst=" + str(target or source)
            + (",readonly" if readonly else "")]


def environment(values):
    return [part for key, value in values.items() for part in ("--env", f"{key}={value}")]


def health(container, port):
    program = (
        "import urllib.request; "
        f"urllib.request.urlopen('http://127.0.0.1:{port}/healthz', timeout=3)"
    )
    for _ in range(60):
        result = subprocess.run(
            ["docker", "exec", container, "python", "-c", program], capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("Synthetic service did not become healthy: " + container)


def launch(root, evaluator, hermes):
    script = Path(__file__).with_name("http_check.py").resolve()
    spec = importlib.util.spec_from_file_location("http_check", script)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    directory = root / "operator/http-acceptance"
    check.prepare(directory, root / "operator/metric")
    for name in ("state", "usage", "traces"):
        (directory / name).mkdir()
    (directory / "token").write_text("sc35-synthetic-submit-token\n")
    (directory / "key").write_text("sc35-synthetic-model-key\n")
    workers, control = PREFIX + "-workers", PREFIX + "-control"
    names = {key: PREFIX + "-" + key for key in ("mock", "gateway", "evaluator")}
    existing = set(docker("ps", "--all", "--format", "{{.Names}}").splitlines())
    if existing.intersection(names.values()):
        raise ValueError("Previous synthetic containers exist; inspect before another run")
    volume = PREFIX + "-journal"
    docker("volume", "create", volume)
    docker(
        "run", "--rm", "--user", "0:0", "--entrypoint", "python", *mount(directory),
        "--mount", f"type=volume,src={volume},dst=/journal", evaluator, "-c",
        "import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "[os.chown(p/n,10001,10001) for n in ('state','usage','traces')]; "
        "os.chown('/journal',int(sys.argv[2]),int(sys.argv[3]))",
        directory, os.getuid(), os.getgid(),
    )
    networks = []
    started = []
    safe = ["--user", "10001:10001", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:mode=1777,size=2g"]
    client_safe = ["--user", f"{os.getuid()}:{os.getgid()}", *safe[2:]]
    try:
        for network in (workers, control):
            docker("network", "create", "--internal", network)
            networks.append(network)
        docker(
            "run", "-d", "--name", names["mock"], "--network", control,
            *safe, *mount(script, "/http_check.py", True),
            *mount(directory / "answer.json", "/answer.json", True),
            "--entrypoint", "python", evaluator, "/http_check.py", "mock-upstream",
            "--answer", "/answer.json",
        )
        started.append(names["mock"])
        common = {"MODEL_NAME": check.MODEL, "MODEL_TIMEOUT": 21600,
                  "MODEL_RETRY_DELAY": 30, "MODEL_USAGE_DIR": str(directory / "usage"),
                  "MODEL_TRACE_DIR": str(directory / "traces"),
                  "MODEL_PARAMETERS_JSON": '{"reasoning_effort":"max"}',
                  "MODEL_API_KEY_FILE": "/synthetic-key",
                  "MODEL_BASE_URL": f"http://{names['mock']}:8000/v1"}
        docker(
            "run", "-d", "--name", names["gateway"], "--network", control, *safe,
            *mount(directory / "usage"), *mount(directory / "traces"),
            *mount(directory / "key", "/synthetic-key", True), *environment(common),
            evaluator, "python", "-m", "evaluator.gateway",
        )
        started.append(names["gateway"])
        docker("network", "connect", workers, names["gateway"])
        health(names["gateway"], 8081)
        settings = {
            "ARENA_DATASET_PATH": str(directory / "dataset.jsonl"),
            "ARENA_METRIC_PATH": "/private/metric", "ARENA_STATE_DIR": str(directory / "state"),
            "ARENA_TOKEN_FILE": str(directory / "token"), "ARENA_MODEL": check.MODEL,
            "ARENA_MODEL_BASE_URL": f"http://{names['gateway']}:8081/v1",
            "ARENA_RUNNER_MODEL_BASE_URL": f"http://{names['gateway']}:8081/v1",
            "ARENA_RUNNER_NETWORK": workers, "ARENA_RUNNER_IMAGE": hermes,
            "ARENA_RELEASE_ID": check.CAMPAIGN, "ARENA_CAMPAIGN_ID": check.CAMPAIGN,
            "ARENA_CASE_ID": "sc35-preparation-http", "ARENA_DATASET_ID": check.DATASET,
            "ARENA_CONDITIONS_ID": check.CONDITIONS, "ARENA_SPLIT": "public",
            "ARENA_USAGE_DIR": str(directory / "usage"),
            "ARENA_TRACE_DIR": str(directory / "traces"),
            "ARENA_ASSETS_DIR": str(root / "assets"),
            "ARENA_SLOT_DIR": str(root / "operator/state/capacity"),
            "ARENA_EXECUTION_SLOTS": 4, "ARENA_RUNNER_MEMORY": "32g",
            "ARENA_RUNNER_CPUS": 8, "ARENA_RUNNER_PIDS_LIMIT": 2048,
            "ARENA_RUNNER_TMPFS_SIZE": "16g", "ARENA_RUNNER_TMPFS_EXEC": "true",
            "ARENA_PARAMETERS": '{"reasoning_effort":"max"}',
            "ARENA_HARNESSES": json.dumps({"hermes": {
                "image": hermes, "network": workers, "uses_model": True,
            }}),
            "MLFLOW_TRACKING_URI": "http://shared-tracking:5000",
        }
        group = Path("/var/run/docker.sock").stat().st_gid
        docker(
            "run", "-d", "--name", names["evaluator"], "--network", control,
            *safe, "--group-add", group, *mount(directory),
            *mount(root / "operator/metric", "/private/metric", True),
            *mount(root / "assets", readonly=True), *mount(root / "operator/state/capacity"),
            *mount("/var/run/docker.sock"), *environment(settings), evaluator,
            "python", "-m", "evaluator.train", "serve",
        )
        started.append(names["evaluator"])
        docker("network", "connect", "sc35-first-control", names["evaluator"])
        health(names["evaluator"], 8080)
        result = docker(
            "run", "--rm", "--network", control, *client_safe,
            *mount(script, "/http_check.py", True), *mount(directory),
            "--mount", f"type=volume,src={volume},dst=/journal", "--entrypoint", "python",
            evaluator, "/http_check.py", "run", "--directory", directory,
            "--journal", "/journal/search.sqlite3", "--url", f"http://{names['evaluator']}:8080",
            "--token-file", directory / "token",
        )
        check.save(directory / "summary.json", json.loads(result))
        payload = docker(
            "run", "--rm", "--network", "sc35-first-researchers", *safe,
            *mount(script, "/http_check.py", True),
            *mount(directory / "receipt.json", "/receipt.json", True),
            "--entrypoint", "python", evaluator, "/http_check.py", "read-trace",
            "--receipt", "/receipt.json", "--uri", "http://shared-tracking:5000",
        )
        result = json.loads(payload)
        check.save(directory / "summary.json", result)
        return result
    finally:
        for name in started:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=30,
            )
            (directory / (name + ".log")).write_text(logs.stdout + logs.stderr)
        if started:
            docker("rm", "--force", *started, check=False)
        for network in reversed(networks):
            docker("network", "rm", network, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluator-image", required=True)
    parser.add_argument("--hermes-image", required=True)
    args = parser.parse_args()
    print(json.dumps(launch(args.root, args.evaluator_image, args.hermes_image), indent=2))


if __name__ == "__main__":
    main()
