import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from evaluator.candidates import materialize_candidate
from evaluator.client import pack_candidate
from evaluator.reliability import empty_usage
from evaluator.runner import DockerRunner, HarnessRunner, RunnerError
from evaluator.search import Journal, Resources
from evaluator.usage import create_usage, seal_usage

PROGRAM = """import json
import os
import sys
import time
from pathlib import Path
from helpers import solve_numbers

task = json.load(sys.stdin)
assert set(task) == {"input", "assets_dir", "parameters"}
assert task["assets_dir"] == "/assets"
assert set(os.listdir("/sys/class/net")) == {"lo"}
assert not any("MODEL" in key or "TOKEN" in key or "API_KEY" in key for key in os.environ)
for path in ("/usage", "/private", "/state", "/run/secrets", "/var/run/docker.sock"):
    assert not Path(path).exists(), path
parameters = task["parameters"]
answer = solve_numbers(task["input"], parameters)
time.sleep(0.05)
print(json.dumps({"answer": "" if parameters["blank"] else answer, "usage": {"total_tokens": 999}}))
"""

PREPARE = """import json
import os
import sys
import uuid
from pathlib import Path
from context_helpers import context
from urllib.parse import urlsplit

task = json.load(sys.stdin)
url = urlsplit(task["model_base_url"])
_, prefix, run_id, credential, version = url.path.split("/")
assert url.netloc == "model-gateway:8081" and prefix == "runs" and version == "v1"
assert str(uuid.UUID(run_id)) == run_id
assert len(credential) >= 32
assert task["input"] in {"synthetic task", "", " \\n\\t"}
assert task == {
    "input": task["input"],
    "assets_dir": "/assets",
    "model": "fake-model",
    "model_base_url": f"http://model-gateway:8081/runs/{run_id}/{credential}/v1",
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
print(json.dumps({"context": context()}))
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
                {"role": "user", "content": body["messages"][-1]["content"]},
            ],
        }
        valid = (
            self.path == "/v1/chat/completions"
            and self.headers.get("Authorization") == "Bearer synthetic-key"
            and body["messages"][-1]["content"] in {"synthetic task", "", " \\n\\t"}
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
        print(json.dumps({"valid": valid, "input": body["messages"][-1]["content"]}), flush=True)

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


def check_program(image: str, workspace: Path) -> None:
    candidate = workspace / "program"
    candidate.mkdir()
    (candidate / "solve.py").write_text(PROGRAM)
    (candidate / "helpers.py").write_text(
        "from formatting.numbers import format_numbers\n"
        "def solve_numbers(text, parameters):\n"
        "    numbers = sorted(map(int, text.split()), reverse=parameters['descending'])\n"
        "    return format_numbers(numbers, parameters['separator'])\n"
    )
    (candidate / "formatting").mkdir()
    (candidate / "formatting" / "__init__.py").write_text("")
    (candidate / "formatting" / "numbers.py").write_text(
        "def format_numbers(numbers, separator):\n"
        "    return separator.join(map(str, numbers))\n"
    )
    runner = HarnessRunner(
        DockerRunner(image),
        {"plain": DockerRunner(image, uses_model=False, model_base_url="", timeout=15)},
    )
    for blank, expected in ((False, "3|2|1"), (True, "")):
        (candidate / "solution.json").write_text(json.dumps({
            "harness": "plain",
            "parameters": {"descending": True, "separator": "|", "blank": blank},
        }))
        started = time.monotonic()
        solved = runner(candidate, "3 1 2")
        elapsed = time.monotonic() - started
        assert solved.answer == expected
        assert solved.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "attempts": 0,
            "unreported_attempts": 0,
        }
        assert 0.05 <= solved.wall_seconds <= elapsed
        assert not (workspace / "usage").exists()


def check_restored_program(image: str, workspace: Path) -> None:
    source = workspace / "executable-source"
    source.mkdir()
    (source / "solve.py").write_text(
        "import json\nfrom helpers.execution import answer\n"
        "print(json.dumps({'answer': answer()}))\n"
    )
    (source / "helpers").mkdir()
    (source / "helpers" / "__init__.py").write_text("")
    (source / "helpers" / "execution.py").write_text(
        "import subprocess\nfrom pathlib import Path\n"
        "def answer():\n"
        "    helper = Path(__file__).parent / 'helper.sh'\n"
        "    return subprocess.check_output([str(helper)], text=True).strip()\n"
    )
    (source / "helpers" / "execution.py").chmod(0o600)
    (source / "empty").mkdir()
    helper = source / "helpers" / "helper.sh"
    helper.write_text("#!/bin/sh\nprintf 'executable helper\\n'\n")
    helper.chmod(0o700)
    journal = Journal.create(workspace / "search.sqlite", max_solutions=1)
    proposed = journal.propose(
        "executable", source, hypothesis="Executable files survive transport",
        change="Use a program with an executable helper", resources=Resources(0, 0),
    )
    restored = workspace / "restored"
    journal.restore("executable", restored)
    packed = pack_candidate("executable", restored)
    assert proposed["digest"] == packed.digest()
    private = workspace / "materialized"
    materialize_candidate(packed, private)
    runner = DockerRunner(image, uses_model=False, model_base_url="", timeout=15)
    for candidate in (restored, private):
        assert pack_candidate("executable", candidate).digest() == packed.digest()
        assert candidate.stat().st_mode & 0o777 == 0o755
        assert not (candidate / "empty").exists()
        assert runner(candidate, "").answer == "executable helper"


def check_accounting_scope(image: str, network: str, usage: Path) -> None:
    victim = str(uuid.uuid4())
    attacker = str(uuid.uuid4())
    create_usage(usage, victim)
    credential = create_usage(usage, attacker)
    code = """import json
import socket
import sys
import urllib.error
import urllib.request

victim, credential = sys.argv[1:]
payload = json.dumps({
    "model": "fake-model", "messages": [{"role": "user", "content": "x"}],
}).encode()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for scope in (victim, victim + "/wrong", victim + "/" + credential):
    request = urllib.request.Request(
        "http://model-gateway:8081/runs/" + scope + "/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        opener.open(request, timeout=3)
    except urllib.error.HTTPError as error:
        assert error.code in {403, 404}, error.code
    else:
        raise AssertionError("another candidate's accounting scope was accepted")
with socket.create_connection(("model-gateway", 8081), timeout=3) as connection:
    headers = (
        f"POST /runs/{victim}/{credential}/v1/chat/completions HTTP/1.1\\r\\n"
        "Host: model-gateway\\r\\nContent-Type: application/json\\r\\n"
        "Content-Length: 10000\\r\\n\\r\\n"
    )
    connection.sendall(headers.encode())
    assert connection.recv(4096).split(b" ")[1] in {b"403", b"404"}
"""
    docker(
        "run", "--rm", "--network", network, "--entrypoint", "python", image,
        "-I", "-c", code, victim, credential,
    )
    assert seal_usage(usage, victim) == empty_usage()
    assert seal_usage(usage, attacker) == empty_usage()


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
        check_program(image, workspace)
        check_restored_program(image, workspace)
        candidate = workspace / "candidate"
        candidate.mkdir()
        (candidate / "prompt.md").write_text("synthetic prompt")
        (candidate / "prepare.py").write_text(PREPARE)
        (candidate / "context_helpers").mkdir()
        (candidate / "context_helpers" / "__init__.py").write_text(
            "from pathlib import Path\n"
            "def context():\n"
            "    return Path('/assets/context.txt').read_text()\n"
        )
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
            check_accounting_scope(image, network, usage)
            assert docker("logs", upstream) == ""
            inputs = ("synthetic task", "", " \n\t")
            for input_text in inputs:
                solved = runner(candidate, input_text)
                assert solved.answer == "synthetic answer"
                assert solved.usage == {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "attempts": 1,
                    "unreported_attempts": 0,
                }
                assert solved.wall_seconds > 0
            expected_log = [{"valid": True, "input": input_text} for input_text in inputs]
            actual_log = [json.loads(line) for line in docker("logs", upstream).splitlines()]
            assert actual_log == expected_log
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
            actual_log = [json.loads(line) for line in docker("logs", upstream).splitlines()]
            assert actual_log == expected_log
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
                {
                    "status": "ok",
                    "runner_image": image,
                    "gateway_image": args.gateway_image,
                    "model_free_solves": 4,
                    "model_solves": 3,
                    "accounting_attacks_rejected": 4,
                }
            )
        )


if __name__ == "__main__":
    main()
