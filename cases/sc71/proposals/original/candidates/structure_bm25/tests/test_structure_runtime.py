"""Real subprocess and loopback HTTP tests; no laboratory model is contacted."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2]


@pytest.fixture
def service():
    calls = []
    state = {"bad_plan": False, "bad_answer": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            if payload["messages"][0]["content"].startswith("Составь план"):
                text = (
                    "not JSON"
                    if state["bad_plan"]
                    else json.dumps(
                        {"subqueries": ["Согласование приобретения"]}, ensure_ascii=False
                    )
                )
            else:
                data = json.loads(payload["messages"][1]["content"])
                evidence = data["evidence"][0]
                text = f"Подтверждение: «{evidence['text'].strip()}» {evidence['citation']}"
                if state["bad_answer"]:
                    text = "Ошибка источника [/kb/norm/" + "f" * 64 + ".md, 1]"
            body = json.dumps(
                {
                    "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", calls, state
    server.shutdown()
    server.server_close()
    thread.join()


def prepare(tmp_path, name, url):
    # Running a copied package catches accidental imports from the original workspace.
    sys.path.insert(0, str(PACKAGES.parent))
    from scripts.export import export
    bundle = tmp_path / "portable"
    manifest = export(name, bundle)
    assert isinstance(manifest["files"], list)
    assert all((bundle / name).is_file() for name in manifest["files"])
    package = bundle / "candidates" / name
    kb = tmp_path / "norm"
    kb.mkdir()
    (kb / ("a" * 64 + ".md")).write_text(
        "# ПРИКАЗ\n1. Комиссия оформляет заявку на оборудование.\n", encoding="utf-8"
    )
    (kb / ("b" * 64 + ".md")).write_text(
        "# ПОЛОЖЕНИЕ\n1. Согласование приобретения осуществляет директор.\n", encoding="utf-8"
    )
    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        SC71_CHAT_URL=url,
        SC71_CHAT_MODEL="synthetic-test",
        SC71_CHAT_KEY="synthetic-key",
    )
    index = tmp_path / "index"
    build = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(package / "build.py"),
            "--kb-dir",
            str(kb),
            "--build-dir",
            str(index),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        check=True,
    )
    assert json.loads(build.stdout)["documents"] == 2
    return package, index, env


@pytest.mark.parametrize("name,expected_calls", [("structure_bm25", 1)])
def test_copied_package_full_answer_and_read_only_execution(
    tmp_path, service, name, expected_calls
):
    url, calls, _ = service
    package, index, env = prepare(tmp_path, name, url)
    before = {
        p: p.read_bytes() for folder in (package, index) for p in folder.rglob("*") if p.is_file()
    }
    run = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(package / "run.py"),
            "--build-dir",
            str(index),
            "--question",
            "Кто оформляет заявку на оборудование?",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        check=True,
    )
    assert "/kb/norm/" in run.stdout
    trace = json.loads(run.stderr)
    assert trace["model_calls_generation"] == 1
    assert len(calls) == expected_calls
    assert "synthetic-key" not in run.stdout + run.stderr
    assert {
        p: p.read_bytes() for folder in (package, index) for p in folder.rglob("*") if p.is_file()
    } == before
    if name == "planned_bm25":
        assert trace["model_calls_retrieval"] == 1
        assert trace["planner"]["usage"]["total_tokens"] == 18
        assert calls[0]["model"] == calls[1]["model"] == "synthetic-test"


def test_invalid_generated_answer_is_retained_and_flagged(tmp_path, service):
    url, calls, state = service
    package, index, env = prepare(tmp_path, "structure_bm25", url)
    state["bad_answer"] = True
    run = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(package / "run.py"),
            "--build-dir",
            str(index),
            "--question",
            "Кто оформляет заявку?",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
    )
    assert run.returncode == 2 and "Ошибка источника" in run.stdout
    assert json.loads(run.stderr)["status"] == "answer_validation_failed"
    assert len(calls) == 1
