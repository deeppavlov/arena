import argparse
import http.client
import json
import socket
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SIZE = 160 * 1024 * 1024
CHUNK = 1024 * 1024
PATTERN = bytes(range(256)) * (CHUNK // 256 + 1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class Backend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        size = SIZE if self.path == "/artifact" else 0
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        for _ in range(size // CHUNK):
            self.wfile.write(PATTERN[:CHUNK])

    def do_PUT(self) -> None:
        require(self.headers.get("Host") == "localhost:5050", "upstream Host changed")
        require(not self.headers.get("Authorization"), "Authorization reached upstream")
        require(self.headers.get("Transfer-Encoding") == "chunked", "upload was buffered")
        total = 0
        while True:
            remaining = int(self.rfile.readline().strip(), 16)
            if remaining == 0:
                require(self.rfile.readline() == b"\r\n", "invalid upload terminator")
                break
            while remaining:
                part = self.rfile.read(min(CHUNK, remaining))
                require(bool(part), "truncated upload")
                offset = total % 256
                require(part == PATTERN[offset:offset + len(part)], "upload content differs")
                total += len(part)
                remaining -= len(part)
            require(self.rfile.read(2) == b"\r\n", "invalid upload chunk")
        require(total == SIZE, "upload size differs")
        body = json.dumps({"upload_bytes": total}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def client() -> None:
    deadline = time.monotonic() + 20
    while True:
        connection = http.client.HTTPConnection("proxy", 5000, timeout=60)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                break
        except OSError:
            pass
        connection.close()
        require(time.monotonic() < deadline, "proxy health failed")
        time.sleep(0.1)
    connection.request(
        "PUT", "/artifact", (PATTERN[:CHUNK] for _ in range(SIZE // CHUNK)),
        {"Authorization": "Bearer synthetic-only"}, encode_chunked=True,
    )
    response = connection.getresponse()
    require(response.status == 200, "upload failed")
    require(json.loads(response.read()) == {"upload_bytes": SIZE}, "upload receipt differs")
    connection.close()
    connection = http.client.HTTPConnection("proxy", 5000, timeout=60)
    connection.connect()
    require(connection.sock is not None, "download socket unavailable")
    if connection.sock is not None:
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    connection.request("GET", "/artifact")
    response = connection.getresponse()
    require(response.status == 200, "download failed")
    time.sleep(3)
    if connection.sock is not None:
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * CHUNK)
    total = 0
    while part := response.read(CHUNK):
        offset = total % 256
        require(part == PATTERN[offset:offset + len(part)], "download content differs")
        total += len(part)
    connection.close()
    require(total == SIZE, "download size differs")
    print(json.dumps({"upload_bytes": SIZE, "download_bytes": total, "content_equal": True}))


def docker(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, check=check, timeout=180,
    )
    return result.stdout.strip()


def memory_events(container: str) -> dict[str, int]:
    output = docker("exec", container, "sh", "-c", """
if [ -f /sys/fs/cgroup/memory.events ]; then
    cat /sys/fs/cgroup/memory.events
else
    for root in /sys/fs/cgroup/memory /sys/fs/cgroup; do
        if [ -f "$root/memory.oom_control" ]; then
            cat "$root/memory.oom_control"
            printf 'failcnt '
            cat "$root/memory.failcnt"
            exit 0
        fi
    done
    exit 1
fi
""")
    return {key: int(value) for key, value in (row.split() for row in output.splitlines())}


def smoke(config: Path, nginx_image: str, python_image: str) -> None:
    name = "arena-mlflow-smoke-" + uuid.uuid4().hex[:12]
    backend, proxy = name + "-backend", name + "-proxy"
    started = time.monotonic()
    require(config.is_file(), "nginx configuration is missing")
    try:
        docker("network", "create", "--internal", name)
        docker(
            "run", "-d", "--pull", "never", "--name", backend,
            "--network", name, "--network-alias", "operator-source", "--read-only",
            "--user", "10001:10001", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--memory", "128m",
            "--memory-swap", "128m", "--pids-limit", "64",
            "-v", f"{Path(__file__).resolve()}:/smoke.py:ro",
            python_image, "python", "/smoke.py", "--serve",
        )
        docker(
            "run", "-d", "--pull", "never", "--name", proxy,
            "--network", name, "--network-alias", "proxy", "--read-only",
            "--user", "10001:10001", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--memory", "128m",
            "--memory-swap", "128m", "--pids-limit", "64",
            "--tmpfs", "/tmp:mode=1777,size=256m",
            "-v", f"{config.resolve()}:/etc/nginx/nginx.conf:ro",
            "--entrypoint", "nginx", nginx_image, "-g", "daemon off;",
        )
        docker("exec", proxy, "nginx", "-t")
        transfer = json.loads(docker("exec", backend, "python", "/smoke.py", "--client"))
        state = json.loads(docker("inspect", proxy))[0]
        require(state["State"]["Running"], "nginx stopped")
        require(not state["State"]["OOMKilled"], "nginx was OOM-killed")
        require(state["RestartCount"] == 0, "nginx restarted")
        events = memory_events(proxy)
        require(all(events.get(key, 0) == 0
                    for key in ("max", "oom", "oom_kill", "oom_group_kill", "failcnt")),
                "nginx exceeded its memory limit")
        print(json.dumps({
            "status": "passed", **transfer, "memory_limit_bytes": 128 * 1024 * 1024,
            "memory_events": events,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }))
    except subprocess.CalledProcessError as error:
        try:
            events = memory_events(proxy)
        except subprocess.CalledProcessError:
            events = {}
        logs = subprocess.run(["docker", "logs", backend], capture_output=True, text=True,
                              check=False, timeout=10)
        containers = json.loads(docker("inspect", proxy, check=False) or "[]")
        print(json.dumps({
            "status": "failed", "command": error.cmd,
            "error": (error.stdout + error.stderr)[-2000:],
            "backend_log": (logs.stdout + logs.stderr)[-2000:],
            "memory_events": events,
            "nginx_state": containers[0]["State"] if containers else None,
        }))
        raise SystemExit(1) from None
    finally:
        docker("rm", "-f", proxy, backend, check=False)
        docker("network", "rm", name, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic large-body nginx transport regression")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "deploy/mlflow/operator-nginx.conf")
    parser.add_argument("--nginx-image", default="nginx:1.28-alpine")
    parser.add_argument("--python-image", default="python:3.12-slim")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--client", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        ThreadingHTTPServer(("0.0.0.0", 5000), Backend).serve_forever()
    elif args.client:
        client()
    else:
        smoke(args.config, args.nginx_image, args.python_image)


if __name__ == "__main__":
    main()
