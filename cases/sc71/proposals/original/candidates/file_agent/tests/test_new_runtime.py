"""Copied packages, real subprocesses and loopback HTTP; laboratory services unused."""

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
    calls, state = [], {"invalid_answer": False, "invalid_agent": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            if "Инструкция агенту поиска" in payload["messages"][0]["content"]:
                text = (
                    "bad JSON"
                    if state["invalid_agent"]
                    else json.dumps({"tool": "finish", "args": {}})
                )
            else:
                data = json.loads(payload["messages"][1]["content"])
                hit = data["evidence"][0]
                text = f"Основание: «{hit['text'].strip()}» {hit['citation']}"
                if state["invalid_answer"]:
                    text = "Недостоверный ответ [/kb/norm/" + "f" * 64 + ".md, 1]"
            data = {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", calls, state
    server.shutdown()
    server.server_close()
    thread.join()


def snapshot(folder):
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in folder.rglob("*")
        if p.is_file()
    }


def prepare(tmp_path, name, url):
    sys.path.insert(0, str(PACKAGES.parent))
    from scripts.export import export
    bundle = tmp_path / "portable"
    export(name, bundle)
    package = bundle / "candidates" / name
    kb = tmp_path / "norm"
    kb.mkdir()
    (kb / ("a" * 64 + ".md")).write_text(
        "# ПРИКАЗ\n01.01.2024 № 10-1\nО закупках\n1. Комиссия согласует закупку.\n",
        encoding="utf-8",
    )
    (kb / ("b" * 64 + ".md")).write_text(
        (
            "# ПРИКАЗ\n02.02.2025 № 20-1\nОб изменении\n1. Внести изменение в приказ о"
            "т 01.01.2024 № 10-1: пункт 1 изложить в следующей редакции: «Ректор со"
            "гласует закупку».\n"
        ),
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        SC71_CHAT_URL=url,
        SC71_CHAT_MODEL="synthetic",
        SC71_CHAT_KEY="synthetic-secret",
    )
    index = tmp_path / "index"
    built = subprocess.run(
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
        env=env,
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert json.loads(built.stdout)["catalog"]["identified"] == 2
    return package, index, env


@pytest.mark.parametrize("name,expected_calls", [("file_agent", 2), ("amendment_bm25", 1)])
def test_full_answer_from_copied_package_is_read_only(tmp_path, service, name, expected_calls):
    url, calls, _ = service
    package, index, env = prepare(tmp_path, name, url)
    before = snapshot(tmp_path)
    run = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(package / "run.py"),
            "--build-dir",
            str(index),
            "--question",
            "Кто согласует закупку?",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert "/kb/norm/" in run.stdout
    assert len(calls) == expected_calls
    trace = json.loads(run.stderr)
    assert trace["model_calls_generation"] == 1
    assert trace["usage_generation"]["total_tokens"] == 18
    assert "synthetic-secret" not in run.stdout + run.stderr
    assert snapshot(tmp_path) == before
    assert all(x["model"] == "synthetic" for x in calls)
    # The common final generator receives only source evidence and relation metadata.
    final = json.loads(calls[-1]["messages"][1]["content"])
    assert set(final) == {"question", "evidence", "references"}
    assert "agent" not in final and "ground_truth" not in json.dumps(final)


@pytest.mark.parametrize("name", ["file_agent", "amendment_bm25"])
def test_invalid_generated_answer_is_retained(tmp_path, service, name):
    url, calls, state = service
    package, index, env = prepare(tmp_path, name, url)
    state["invalid_answer"] = True
    run = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(package / "run.py"),
            "--build-dir",
            str(index),
            "--question",
            "Кто согласует закупку?",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert run.returncode == 2 and "Недостоверный ответ" in run.stdout
    assert json.loads(run.stderr)["status"] == "answer_validation_failed"
    assert len(calls) == (2 if name == "file_agent" else 1)


def test_malformed_agent_reply_has_partial_trace_and_no_generation(tmp_path, service):
    url, calls, state = service
    package, index, env = prepare(tmp_path, "file_agent", url)
    state["invalid_agent"] = True
    run = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(package / "run.py"),
            "--build-dir",
            str(index),
            "--format",
            "json",
            "--question",
            "Кто согласует закупку?",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert run.returncode == 2 and not run.stdout
    error = json.loads(run.stderr)
    assert error["diagnostics"]["model_calls_retrieval"] == 1
    assert len(calls) == 1 and "synthetic-secret" not in run.stderr


@pytest.mark.parametrize(
    "name,flag", [("file_agent", "--no-agent"), ("amendment_bm25", "--no-relations")]
)
def test_ablation_uses_no_model(tmp_path, service, name, flag):
    url, calls, _ = service
    package, index, env = prepare(tmp_path, name, url)
    command = [
        sys.executable,
        "-S",
        "-B",
        str(package / "run.py"),
        "--build-dir",
        str(index),
        "--question",
        "Кто согласует закупку?",
        "--format",
        "json",
        flag,
    ]
    run = subprocess.run(
        command, env=env, cwd=tmp_path, text=True, encoding="utf-8", capture_output=True
    )
    assert run.returncode == 0 and json.loads(run.stdout)["model_calls_retrieval"] == 0
    assert calls == []
