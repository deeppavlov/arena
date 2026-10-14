import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from evaluator.runner import DockerRunner, RunnerError

PREPARE = """import json
import os
import sys
import uuid
from pathlib import Path

task = json.load(sys.stdin)
run_id = task["model_base_url"].removeprefix("http://model-gateway:8081/runs/").removesuffix("/v1")
assert str(uuid.UUID(run_id)) == run_id
assert task == {
    "input": "synthetic task",
    "assets_dir": "/assets",
    "model": "fake-model",
    "model_base_url": f"http://model-gateway:8081/runs/{run_id}/v1",
    "model_request_timeout": 72,
}
assert os.getuid() == 10001 and os.getgid() == 10001
status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
assert int(status["CapEff"].strip(), 16) == 0
assert status["NoNewPrivs"].strip() == "1"
mounts = {
    fields[4]: set(fields[5].split(","))
    for fields in (line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines())
}
for path in ("/", "/candidate", "/assets"):
    assert "ro" in mounts[path], path
assert {"rw", "noexec", "nosuid", "nodev"} <= mounts["/tmp"]
for path in ("/private", "/state", "/usage", "/run/secrets", "/var/run/docker.sock"):
    assert not Path(path).exists(), path
assert not any("TOKEN" in key or "API_KEY" in key for key in os.environ)
for path in ("/candidate/write", "/assets/write", "/opt/runner/write"):
    try:
        Path(path).write_text("blocked")
    except OSError:
        pass
    else:
        raise AssertionError(path)
Path("/tmp/write").write_text("allowed")
assert Path("/assets/context.txt").read_text() == "public context"
print(json.dumps({"context": "public context"}))
"""

UPSTREAM = """import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        expected = {
            "model": "fake-model",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "synthetic prompt"},
                {"role": "user", "content": "Task context:\\npublic context"},
                {"role": "user", "content": "synthetic task"},
            ],
        }
        valid = (
            self.path == "/v1/chat/completions"
            and self.headers.get("Authorization") == "Bearer synthetic-key"
            and body == expected
        )
        output = json.dumps({
            "choices": [{"message": {"content": "synthetic answer"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }).encode()
        self.send_response(200 if valid else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(output)))
        self.end_headers()
        self.wfile.write(output)
        print("valid request" if valid else "invalid request", flush=True)

HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
"""


def docker(*arguments: str, check: bool = True) -> str:
    return subprocess.run(
        ["docker", *arguments],
        capture_output=True,
        text=True,
        check=check,
        timeout=180,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-image", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suffix = uuid.uuid4().hex[:12]
    network = f"arena-smoke-{suffix}"
    upstream = f"arena-smoke-upstream-{suffix}"
    gateway = f"arena-smoke-gateway-{suffix}"
    with tempfile.TemporaryDirectory(prefix="arena-smoke-") as temporary:
        workspace = Path(temporary)
        image_file = workspace / "image-id"
        subprocess.run(
            ["docker", "build", "-f", "runner/Dockerfile", "--iidfile", str(image_file), "."],
            cwd=root,
            check=True,
            timeout=180,
        )
        image = image_file.read_text().strip()
        candidate = workspace / "candidate"
        candidate.mkdir()
        (candidate / "prompt.md").write_text("synthetic prompt")
        (candidate / "prepare.py").write_text(PREPARE)
        assets = workspace / "assets"
        assets.mkdir()
        (assets / "context.txt").write_text("public context")
        usage = workspace / "usage"
        usage.mkdir(mode=0o700)
        key = workspace / "synthetic-key"
        key.write_text("synthetic-key")
        key.chmod(0o644)
        network_created = False
        try:
            docker("network", "create", "--internal", network)
            network_created = True
            assert json.loads(docker("network", "inspect", network))[0]["Internal"]
            docker(
                "run",
                "--detach",
                "--name",
                upstream,
                "--network",
                network,
                "--network-alias",
                "fake-upstream",
                "--entrypoint",
                "python",
                image,
                "-u",
                "-c",
                UPSTREAM,
            )
            docker(
                "run",
                "--detach",
                "--name",
                gateway,
                "--network",
                network,
                "--network-alias",
                "model-gateway",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--env",
                "MODEL_NAME=fake-model",
                "--env",
                "MODEL_BASE_URL=http://fake-upstream:8000/v1",
                "--env",
                'MODEL_PARAMETERS_JSON={"temperature":0}',
                "--env",
                "MODEL_API_KEY_FILE=/run/secrets/model-key",
                "--env",
                "MODEL_TIMEOUT=2",
                "--env",
                "MODEL_RETRY_DELAY=0",
                "--env",
                "MODEL_USAGE_DIR=/usage",
                "--mount",
                f"type=bind,src={key},dst=/run/secrets/model-key,readonly",
                "--mount",
                f"type=bind,src={usage},dst=/usage",
                args.gateway_image,
                "python",
                "-m",
                "evaluator.gateway",
            )
            health = (
                "import urllib.request; "
                "assert urllib.request.urlopen('http://127.0.0.1:8081/healthz').status == 200"
            )
            for _ in range(50):
                result = subprocess.run(
                    ["docker", "exec", gateway, "python", "-c", health],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("gateway did not become healthy")
            runner = DockerRunner(
                image=image,
                network=network,
                model="fake-model",
                assets_dir=assets,
                timeout=15,
                model_timeout=2,
                retry_delay=0,
                usage_dir=usage,
            )
            solved = runner(candidate, "synthetic task")
            assert solved.answer == "synthetic answer"
            assert solved.usage == {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "attempts": 1,
                "unreported_attempts": 0,
            }
            assert solved.wall_seconds > 0
            assert docker("logs", upstream) == "valid request"
            (candidate / "prepare.py").write_text(
                PREPARE.replace('"model": "fake-model"', '"model": "different-model"')
            )
            wrong_model = DockerRunner(
                image=image,
                network=network,
                model="different-model",
                assets_dir=assets,
                timeout=15,
                model_timeout=2,
                retry_delay=0,
                usage_dir=usage,
            )
            try:
                wrong_model(candidate, "synthetic task")
            except RunnerError as error:
                assert str(error) == "runner failed"
            else:
                raise AssertionError("gateway accepted a different model")
            assert docker("logs", upstream) == "valid request"
            (candidate / "prepare.py").write_text("import time\ntime.sleep(30)\n")
            slow_runner = DockerRunner(
                image=image,
                network=network,
                model="fake-model",
                assets_dir=assets,
                timeout=2,
                model_timeout=2,
                retry_delay=0,
                usage_dir=usage,
            )
            try:
                slow_runner(candidate, "synthetic task")
            except RunnerError as error:
                assert str(error) == "runner timed out"
            else:
                raise AssertionError("runner did not time out")
            names = set(
                docker(
                    "ps", "--all", "--filter", f"network={network}", "--format", "{{.Names}}"
                ).splitlines()
            )
            assert names == {upstream, gateway}, names
        finally:
            if network_created:
                containers = docker("ps", "--all", "--quiet", "--filter", f"network={network}")
                if containers:
                    docker("rm", "--force", *containers.splitlines())
                docker("network", "rm", network)
        print(
            json.dumps(
                {"status": "ok", "runner_image": image, "gateway_image": args.gateway_image}
            )
        )


if __name__ == "__main__":
    main()
