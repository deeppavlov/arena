import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "cases" / "template"


def run_metric(path: Path, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        cwd="/",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=5,
    )


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [("Привет 🌍", "Привет 🌍", 1), ("42", "43", 0), (" a", "a", 0), ("", "", 1)],
)
def test_program_metric_example(answer: str, reference: str, expected: int) -> None:
    result = run_metric(
        TEMPLATE / "metric" / "score.py",
        {
            "example": {"id": "synthetic", "input": "Synthetic task", "reference": reference},
            "answer": answer,
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"score": expected}


def test_program_metric_example_requires_reference() -> None:
    result = run_metric(
        TEMPLATE / "metric" / "score.py",
        {"example": {"id": "synthetic", "input": "Synthetic task"}, "answer": "42"},
    )
    assert result.returncode != 0
    assert "reference" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        ([0.25], 0.25),
        ([0.25, 0.75], 0.5),
        ([0.25] * 3 + [0.75] * 4, 3.75 / 7),
        ([True], None),
        ([2], None),
        ([float("nan")], None),
    ],
)
def test_custom_model_metric_uses_configured_votes(
    tmp_path: Path, responses: list[Any], expected: float | None
) -> None:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == "/scope/v1/chat/completions"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            content = json.dumps({"score": responses[len(requests) - 1]})
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    source = TEMPLATE / "metric-examples" / "custom-model"
    (tmp_path / "judge.py").write_bytes((source / "judge.py").read_bytes())
    policy = json.loads((source / "policy.json").read_text())
    policy["votes"] = len(responses)
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            "example": {"id": "synthetic", "input": "Синтетический вопрос", "reference": "A"},
            "answer": "B",
            "model": "fake-internal-model",
            "model_base_url": f"http://127.0.0.1:{server.server_port}/scope/v1",
            "model_request_timeout": 2,
        }
        result = run_metric(tmp_path / "judge.py", payload)
        if expected is None:
            assert result.returncode != 0
            assert "judge score must be" in result.stderr
            assert not result.stdout
        else:
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout) == {
                "score": expected,
                "statistics": {"votes": responses},
            }
        assert len(requests) == len(responses)
        assert all(request == requests[0] for request in requests)
        assert requests[0]["model"] == payload["model"]
        assert json.loads(requests[0]["messages"][1]["content"]) == {
            "example": payload["example"],
            "answer": "B",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_custom_model_metric_has_programmatic_empty_answer_gate() -> None:
    result = run_metric(
        TEMPLATE / "metric-examples" / "custom-model" / "judge.py",
        {"example": {"id": "synthetic", "input": "Synthetic task"}, "answer": "  "},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"score": 0, "statistics": {"votes": []}}
