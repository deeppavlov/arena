import argparse
import gc
import importlib
import json
import os
import resource
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
SDK_PAYLOADS: list[tuple[Any, Any]] = [
    ('{"key":1}', {"key": 1}),
    ('[1,2]', [1, 2]),
    ('"hello"', "hello"),
    ('"{\\"key\\":1}"', '{"key":1}'),
    ("null", "null"),
    ("true", "true"),
    ("false", "false"),
    ("123", "123"),
    (True, True),
    (1, 1),
    (1.0, 1.0),
    ([True, 1, 1.0], [True, 1, 1.0]),
    ("plain text", "plain text"),
    ("", ""),
    (None, None),
    ({"nested": '{"key":1}'}, {"nested": '{"key":1}'}),
    ("data:text/plain;base64,aGVsbG8=", "data:text/plain;base64,aGVsbG8="),
    ({"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
     {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}),
    ({"type": "image", "source": {
        "type": "base64", "data": "aGVsbG8=", "media_type": "image/png",
    }}, {"type": "image", "source": {
        "type": "base64", "data": "aGVsbG8=", "media_type": "image/png",
    }}),
    ({"audio": {"data": "aGVsbG8=", "transcript": "synthetic"}},
     {"audio": {"data": "aGVsbG8=", "transcript": "synthetic"}}),
    ({"inline_data": {"mime_type": "text/plain", "data": "aGVsbG8="}},
     {"inline_data": {"mime_type": "text/plain", "data": "aGVsbG8="}}),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_json_equal(value: Any, expected: Any, message: str) -> None:
    require(json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True), message)


def docker(*arguments: str, check: bool = True, timeout: int = 180) -> str:
    return subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, check=check, timeout=timeout,
    ).stdout.strip()


def make_record(kind: str) -> dict[str, Any]:
    start = time.time_ns() - 10_000_000_000
    count, size = {"aggregate": (320, MIB // 2), "singleton": (1, 280 * MIB),
                   "sdk_payloads": (65, 0)}[kind]
    pattern = "synthetic / value with unicode: я; quote: \"; slash: \\; "
    payload = (pattern * (size // len(pattern) + 1))[:size]
    calls = [{
        "name": f"synthetic-{index}", "span_id": f"source-{index}",
        "parent_id": f"source-{index - 1}" if index else None,
        "span_type": "LLM", "inputs": SDK_PAYLOADS[index % len(SDK_PAYLOADS)][0]
        if kind == "sdk_payloads" else payload,
        "outputs": SDK_PAYLOADS[index % len(SDK_PAYLOADS)][0] if kind == "sdk_payloads"
        else {"index": index, "answer": "synthetic answer"},
        "attributes": {"role": "solver", "synthetic": True,
                       "api_key": "synthetic-hidden-key",
                       **({"json_metadata": '{"synthetic":true}'}
                          if kind == "sdk_payloads" else {})},
        "status": "ERROR" if kind == "sdk_payloads" and index % 11 == 0 else "OK",
        "start_time_ns": start + index * 1_000_000,
        "end_time_ns": start + index * 1_000_000 + 500_000,
    } for index in range(count)]
    result = {
        "split": "train", "status": "FINISHED",
        "params": {
            "case_id": "large-export-smoke", "campaign_id": "synthetic-only",
            "candidate_id": kind, "long_metadata": "metadata value " * 600,
        },
        "metrics": {"score": 1.0, "solver_tokens": 0, "judge_tokens": 0},
        "start_time_ns": start, "end_time_ns": time.time_ns(), "calls": calls,
        "examples": [{"id": f"synthetic-{index}", "calls": [call]}
                     for index, call in enumerate(calls)] if kind == "aggregate" else [],
    }
    return result


def trace_ids(client: Any, experiment_id: str, run_id: str) -> set[str]:
    ids = set()
    query = {"locations": [experiment_id], "run_id": run_id,
             "include_spans": False, "max_results": 100}
    while True:
        page = client.search_traces(**query)
        ids.update(trace.info.trace_id for trace in page)
        token = getattr(page, "token", None)
        if not token:
            return ids
        query["page_token"] = token


def artifact_sizes(client: Any, run_id: str, path: str = "") -> list[int]:
    sizes = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            sizes.extend(artifact_sizes(client, run_id, item.path))
        else:
            sizes.append(item.file_size)
    return sizes


def client_case(kind: str, work: Path) -> None:
    work.mkdir(exist_ok=True)
    (work / "tmp").mkdir(exist_ok=True)

    from evaluator.tracking import export_record, sanitize_record
    from evaluator.tracking_artifacts import read_json_artifact

    mlflow = importlib.import_module("mlflow")
    requests = importlib.import_module("requests")
    requests_seen: list[int] = []
    statuses: list[int] = []
    send = requests.sessions.Session.send

    def measured_send(session: Any, request: Any, **kwargs: Any) -> Any:
        length = request.headers.get("Content-Length")
        body = request.body
        if length is not None:
            requests_seen.append(int(length))
        elif isinstance(body, (bytes, str)):
            requests_seen.append(len(body.encode() if isinstance(body, str) else body))
        else:
            require(body is None, "unmeasured HTTP request body")
        response = send(session, request, **kwargs)
        statuses.append(response.status_code)
        return response

    requests.sessions.Session.send = measured_send
    require(mlflow.__version__ == "3.16.0", "unexpected MLflow version")
    started = time.monotonic()
    source = make_record(kind)
    path = work / "record.json"
    with path.open("w") as stream:
        json.dump(source, stream, ensure_ascii=False)
    source_bytes = path.stat().st_size
    if kind != "sdk_payloads":
        require(source_bytes > 256 * MIB, "synthetic source does not exceed proxy limit")
    source_snapshot = path.read_bytes() if kind == "sdk_payloads" else None
    expected = sanitize_record(source)
    del source
    gc.collect()
    queued: dict[str, Any] = {}
    if kind == "sdk_payloads":
        os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] = "true"
        require(export_record(path, work / "queue", defer=True)
                == {"status": "tracking_pending"}, "deferred source was not queued")
        queued = json.loads(next((work / "queue").glob("*.json")).read_text())
        require_json_equal(queued["record"], expected, "queued source differs")
    result = export_record(path, work / "queue")
    require(result["status"] == "exported", "export did not complete")
    run_id = result["run_id"]
    client = mlflow.MlflowClient(tracking_uri="http://proxy:5000")
    run = client.get_run(run_id)
    require(run.info.status == "FINISHED", "run execution status differs")
    require(run.data.tags.get("arena_export_complete") == "true", "export tag is incomplete")
    if kind == "sdk_payloads":
        require(os.environ.get("MLFLOW_TRACE_EXTRACT_ATTACHMENTS") == "true",
                "export changed the attachment extraction environment")
    restored = read_json_artifact(client, run_id, "record.json")
    if kind == "sdk_payloads":
        require_json_equal(restored, expected, "reconstructed record differs")
    else:
        require(restored == expected, "reconstructed record differs")
    del restored
    gc.collect()
    ids_before = trace_ids(client, run.info.experiment_id, run_id)
    require(len(ids_before) == len(expected["calls"]), "trace count differs")
    seen = set()
    query: dict[str, Any] = {
        "locations": [run.info.experiment_id], "run_id": run_id,
        "include_spans": True, "max_results": 1,
    }
    while True:
        page = client.search_traces(**query)
        for trace in page:
            require(len(trace.data.spans) == 1, "call trace is not independent")
            span = trace.data.spans[0]
            source_id = span.attributes.get("source_span_id")
            require(source_id not in seen, "duplicate logical call")
            seen.add(source_id)
            index = int(source_id.removeprefix("source-"))
            call = expected["calls"][index]
            require(span.name == call["name"], "span name differs")
            require(span.span_type == call["span_type"], "span type differs")
            require(span.parent_id is None, "call trace has a native parent")
            require(span.start_time_ns == call["start_time_ns"], "span start time differs")
            require(span.end_time_ns == call["end_time_ns"], "span end time differs")
            require(span.status.status_code.value == call["status"], "span status differs")
            require(span.attributes.get("source_parent_id") == call["parent_id"],
                    "source parent differs")
            require(span.attributes.get("role") == "solver", "source role differs")
            artifact = span.attributes.get("arena_artifact")
            if artifact:
                require(span.inputs == span.outputs == {"arena_artifact": artifact},
                        "artifact reference differs")
                require(span.attributes.get("arena_payload_format") == "json-artifact-v1",
                        "artifact reference format differs")
                restored_call = read_json_artifact(client, run_id, artifact)
                require(restored_call == call, "referenced call differs")
                del restored_call
            else:
                expected_inputs = SDK_PAYLOADS[index % len(SDK_PAYLOADS)][1] \
                    if kind == "sdk_payloads" else call["inputs"]
                expected_outputs = SDK_PAYLOADS[index % len(SDK_PAYLOADS)][1] \
                    if kind == "sdk_payloads" else call["outputs"]
                if kind == "sdk_payloads":
                    require_json_equal(span.inputs, expected_inputs, "span inputs differ")
                    require_json_equal(span.outputs, expected_outputs, "span outputs differ")
                else:
                    require(span.inputs == expected_inputs, "span inputs differ")
                    require(span.outputs == expected_outputs, "span outputs differ")
                attributes = {**call["attributes"]}
                if kind == "sdk_payloads":
                    attributes["json_metadata"] = {"synthetic": True}
                    restored_call = read_json_artifact(
                        client, run_id, f"calls/inline-{index:08d}.json",
                    )
                    require_json_equal(restored_call, call, "original call artifact differs")
                    require_json_equal(
                        {key: span.attributes.get(key) for key in attributes}, attributes,
                        "span attributes differ",
                    )
                else:
                    require(all(span.attributes.get(key) == value
                                for key, value in attributes.items()),
                            "span attributes differ")
        token = getattr(page, "token", None)
        if not token:
            break
        query["page_token"] = token
    require(len(seen) == len(expected["calls"]), "not all calls were read")
    require(export_record(path, work / "queue") == result, "retry changed run identity")
    require(trace_ids(client, run.info.experiment_id, run_id) == ids_before,
            "completed retry changed traces")
    if kind == "sdk_payloads":
        require(path.read_bytes() == source_snapshot, "source file changed")
        final_queue = json.loads(next((work / "queue").glob("*.json")).read_text())
        require_json_equal(final_queue["record"], queued["record"], "queue payload changed")
        require(final_queue["export_id"] == queued["export_id"], "export identity changed")
        require(os.environ.get("MLFLOW_TRACE_EXTRACT_ATTACHMENTS") == "true",
                "retry changed the attachment extraction environment")
    runs = client.search_runs(
        [run.info.experiment_id],
        filter_string=f"tags.arena_export_id = '{run.data.tags['arena_export_id']}'",
    )
    require(len(runs) == 1, "retry duplicated a run")
    sizes = artifact_sizes(client, run_id)
    require(max(sizes) <= 8 * MIB, "artifact part exceeds transport budget")
    require(max(requests_seen) <= 8 * MIB, "HTTP body exceeds transport budget")
    require(all(status < 400 or status == 404 for status in statuses), "an HTTP request failed")
    print(json.dumps({
        "status": "passed", "case": kind, "source_bytes": source_bytes,
        "logical_calls": len(seen), "run_id": run_id, "full_record_equal": True,
        "call_payloads_equal": True, "stable_retry": True,
        "max_http_request_bytes": max(requests_seen), "max_artifact_bytes": max(sizes),
        "http_status_counts": dict(Counter(statuses)),
        "client_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **({"sdk_payload_variants": len(SDK_PAYLOADS), "call_artifacts_exact": len(seen),
            "source_unchanged": True, "queue_payload_unchanged": True,
            "attachment_environment_restored": True} if kind == "sdk_payloads" else {}),
    }))


def memory_state(container: str) -> dict[str, Any]:
    state = json.loads(docker("inspect", container))[0]
    values = docker("exec", container, "sh", "-c", """
if [ -f /sys/fs/cgroup/memory.events ]; then
    cat /sys/fs/cgroup/memory.events
    printf 'current '; cat /sys/fs/cgroup/memory.current
    printf 'peak '; cat /sys/fs/cgroup/memory.peak
else
    for root in /sys/fs/cgroup/memory /sys/fs/cgroup; do
        if [ -f "$root/memory.oom_control" ]; then
            cat "$root/memory.oom_control"
            printf 'failcnt '; cat "$root/memory.failcnt"
            printf 'current '; cat "$root/memory.usage_in_bytes"
            printf 'peak '; cat "$root/memory.max_usage_in_bytes"
            exit 0
        fi
    done
    exit 1
fi
""")
    counters = {key: int(value) for key, value in (line.split() for line in values.splitlines())}
    require(state["State"]["Running"] and not state["State"]["OOMKilled"],
            "a synthetic container stopped or was OOM killed")
    require(state["RestartCount"] == 0, "a synthetic container restarted")
    require(all(counters.get(key, 0) == 0 for key in ("oom", "oom_kill", "oom_group_kill")),
            "a synthetic process was OOM killed")
    return {"memory_limit_bytes": state["HostConfig"]["Memory"], "restarts": 0,
            "oom_killed": False, **counters}


def smoke(image: str, nginx_image: str, cases: list[str]) -> dict[str, Any]:
    name = "arena-export-smoke-" + uuid.uuid4().hex[:12]
    backend, proxy = name + "-backend", name + "-proxy"
    clients = [name + "-" + kind for kind in cases]
    volumes = [name + "-data", *[item + "-work" for item in clients]]
    try:
        docker("network", "create", "--internal", name)
        for volume in volumes:
            docker("volume", "create", volume)
            docker("run", "--rm", "--pull", "never", "--network", "none",
                   "-v", f"{volume}:/work", "--entrypoint", "python", image, "-c",
                   "import os; os.chown('/work', 10001, 10001)")
        common = ["--pull", "never", "--network", name, "--read-only", "--user", "10001:10001",
                  "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                  "--pids-limit", "512", "--tmpfs", "/tmp:mode=1777,size=512m"]
        docker("run", "-d", "--name", backend, *common,
               "--network-alias", "operator-source", "--memory", "4g", "--memory-swap", "4g",
               "-v", f"{volumes[0]}:/mlflow", "-e", "HOME=/tmp",
               "-e", "MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false", "--entrypoint", "mlflow", image,
               "server", "--backend-store-uri", "sqlite:////mlflow/tracking.db",
               "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts",
               "--host", "0.0.0.0", "--port", "5000", "--workers", "1",
               "--allowed-hosts", "localhost:*,127.0.0.1:*,proxy:*,operator-source:*")
        docker("run", "-d", "--name", proxy, *common,
               "--network-alias", "proxy", "--memory", "128m", "--memory-swap", "128m",
               "-v", f"{ROOT / 'deploy/mlflow/operator-nginx.conf'}:/etc/nginx/nginx.conf:ro",
               "--entrypoint", "nginx", nginx_image, "-g", "daemon off;")
        results = []
        for kind, container, volume in zip(cases, clients, volumes[1:], strict=True):
            docker("run", "-d", "--name", container, *common,
                   "--memory", "4g", "--memory-swap", "4g",
                   "-v", f"{volume}:/work", "-v", f"{ROOT}:/source:ro",
                   "-e", "PYTHONPATH=/source", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-e", "TMPDIR=/work/tmp",
                   "-e", "MLFLOW_TRACKING_URI=http://proxy:5000",
                   "-e", "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=false",
                   "-e", "MLFLOW_ENABLE_ASYNC_LOGGING=false",
                   "-e", "MLFLOW_DISABLE_AGENT_HINT=1",
                   "-e", "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false",
                   "--entrypoint", "tail", image, "-f", "/dev/null")
            docker("exec", container, "python", "-c", """
import time, urllib.request
deadline=time.monotonic()+60
while True:
    try:
        urllib.request.urlopen('http://proxy:5000/health',timeout=3).read()
        break
    except Exception:
        if time.monotonic() > deadline: raise
        time.sleep(0.2)
""")
            output = docker("exec", container, "python", "/source/tests/mlflow_export_smoke.py",
                            "--client", kind, "--work", "/work", timeout=1800)
            result = json.loads(output.splitlines()[-1])
            require(result["status"] == "passed", "client acceptance failed")
            result["memory"] = {"backend": memory_state(backend),
                                "proxy": memory_state(proxy), "client": memory_state(container)}
            results.append(result)
            print(json.dumps({"completed_case": kind,
                              "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return {"status": "passed", "synthetic_only": True, "model_calls": 0,
                "mlflow_version": "3.16.0", "workers": 1, "cases": results}
    finally:
        docker("rm", "-f", *clients, proxy, backend, check=False)
        for volume in volumes:
            docker("volume", "rm", volume, check=False)
        docker("network", "rm", name, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real MLflow export synthetic acceptance")
    parser.add_argument("--image", default="ghcr.io/mlflow/mlflow:v3.16.0")
    parser.add_argument("--nginx-image", default="nginx:1.28-alpine")
    cases = ("aggregate", "singleton", "sdk_payloads")
    parser.add_argument("--cases", choices=cases, nargs="+", default=list(cases))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--client", choices=cases, help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, default=Path("/work"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.client:
        client_case(args.client, args.work)
        return
    result = smoke(args.image, args.nginx_image, args.cases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
