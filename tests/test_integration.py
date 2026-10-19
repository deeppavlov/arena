import base64
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluator.reliability import empty_usage
from evaluator.runner import SolveResult
from evaluator.service import Settings, create_app
from evaluator.usage import add_usage, begin_request, end_request


@pytest.mark.parametrize("fails", [False, True])
def test_custom_model_commands_and_aggregation_use_sealed_gateway_usage(tmp_path: Path, fails):
    usage_dir = tmp_path / "usage"
    requests = []

    class Gateway(BaseHTTPRequestHandler):
        def do_POST(self):
            _, prefix, run_id, capability, version, resource, method = self.path.split("/")
            assert (prefix, version, resource, method) == ("runs", "v1", "chat", "completions")
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["model"] == "fixed-model"
            begin_request(usage_dir, run_id, capability)
            try:
                requests.append(payload)
                add_usage(usage_dir, run_id, {
                    "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
                })
                response = json.dumps({"choices": [{"message": {"content": "4"}}]}).encode()
            finally:
                end_request(usage_dir, run_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        script = tmp_path / "judge.py"
        script.write_text(
            "import json, sys, urllib.request\n"
            "task = json.load(sys.stdin)\n"
            "aggregation = 'results' in task\n"
            "for _ in range(1 if aggregation else 5):\n"
            "    payload = json.dumps({'model': task['model'], 'messages': ["
            "{'role': 'user', 'content': 'synthetic'}]}).encode()\n"
            "    request = urllib.request.Request(task['model_base_url'] + '/chat/completions', "
            "data=payload, headers={'Content-Type': 'application/json'})\n"
            "    with urllib.request.urlopen(request, "
            "timeout=task['model_request_timeout']) as r:\n"
            "        answer = json.load(r)['choices'][0]['message']['content']\n"
            + ("if aggregation:\n    raise RuntimeError('synthetic failure')\n" if fails else "")
            + "print(json.dumps({'score': float(answer)} if aggregation else "
            "{'statistics': {'judgments': 5}}))\n"
        )
        executable = Path(sys.executable).resolve()
        command = {
            "argv": [str(executable), str(script)], "uses_model": True,
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (executable, script)
            },
        }
        metric = tmp_path / "metric.json"
        metric.write_text(json.dumps({
            "version": "custom", "kind": "command", "command": command, "aggregation": command,
        }))
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text('{"id":"one","input":"a"}\n{"id":"two","input":"b"}\n')
        settings = Settings(
            dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
            token="synthetic", runner_image="unused", release_id="test", finalists=1,
            uses_model=False, model="fixed-model", model_base_url="http://internal/v1",
            runner_model_base_url=f"http://127.0.0.1:{server.server_port}/v1", usage_dir=usage_dir,
        )
        api = TestClient(create_app(
            settings, solve=lambda *_: SolveResult("answer", empty_usage(), 1.0),
        ), headers={"Authorization": "Bearer synthetic"})
        body = {"candidates": [{"id": "candidate", "files": {
            "program.txt": base64.b64encode(b"synthetic").decode(),
        }}]}
        assert api.post("/v1/evaluations", json=body).status_code == 202
        assert api.get("/v1/evaluations").json() == {"status": "failed" if fails else "complete"}
        details_path = next(settings.state_dir.glob("*/details/candidate.json"))
        details = json.loads(details_path.read_text())
        assert len(requests) == 11
        assert [row["resources"]["judge"]["usage"]["total_tokens"]
                for row in details["examples"]] == [15, 15]
        assert details["aggregation"]["resources"]["judge"]["usage"]["total_tokens"] == 3
        ledgers = [json.loads(path.read_text()) for path in usage_dir.glob("*.json")]
        assert sorted(row["attempts"] for row in ledgers) == [1, 5, 5]
        assert all(row["closed"] and row["active_requests"] == 0 for row in ledgers)
        if not fails:
            record = json.loads(next(settings.state_dir.glob("*/record.json")).read_text())
            assert record["scores"][0]["judge_tokens"] == 33
            assert record["scores"][0]["solver_tokens"] == 0
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
