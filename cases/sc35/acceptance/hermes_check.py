import argparse
import base64
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from cases.sc35.harness.hermes import solve
from cases.sc71.harness.hermes import runtime
from cases.sc71.harness.hermes.mock_check import MockGateway, effective_settings

FACT = "Синтетический документ 😀"
SECRET = "sc35-fixture-capability"


def synthetic_input() -> dict:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    document = ET.Element(namespace + "document")
    body = ET.SubElement(document, namespace + "body")
    paragraph = ET.SubElement(body, namespace + "p")
    run = ET.SubElement(paragraph, namespace + "r")
    ET.SubElement(run, namespace + "t").text = FACT
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", ET.tostring(document))
    data = output.getvalue()
    return {
        "case_id": "synthetic",
        "docx_base64": base64.b64encode(data).decode(),
        "document_sha256": hashlib.sha256(data).hexdigest(),
        "text_sha256": hashlib.sha256((FACT + "\n").encode()).hexdigest(),
    }


class Gateway(MockGateway):
    def __init__(self, scenario, document, answer):
        super().__init__(scenario, document)
        self.answer = answer

    def respond(self, payload):
        self.requests.append(payload)
        if payload.get("model") != runtime.MODEL or payload.get("reasoning_effort") != "max":
            self.invalid_requests += 1
            return 400, {"error": {"message": "Invalid fixture model settings"}}
        if self.scenario == "failure":
            return 500, {"error": {"message": "Synthetic failure"}}
        messages = payload.get("messages", [])
        tool_results = [message for message in messages if message.get("role") == "tool"]
        message = {"role": "assistant", "reasoning_content": "Synthetic reasoning."}
        if tool_results:
            message["content"] = self.answer
        else:
            message["content"] = None
            message["tool_calls"] = [{
                "id": "synthetic-document-inspection",
                "type": "function",
                "function": {"name": "inspect_document", "arguments": "{}"},
            }]
        return 200, {
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "stop" if tool_results else "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def check(scenario):
    with tempfile.TemporaryDirectory(prefix="sc35-hermes-") as directory:
        root = Path(directory)
        candidate, assets, workspace, home = (root / name for name in (
            "candidate", "assets", "workspace", "home"
        ))
        for path in (candidate, assets, workspace):
            path.mkdir()
        (candidate / "prompt.md").write_text(
            "Inspect the synthetic document with inspect_document."
        )
        (candidate / "tools.json").write_text(json.dumps([{
            "name": "inspect_document", "description": "Inspect the current synthetic document",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "script": "inspect.py",
        }]))
        (candidate / "inspect.py").write_text(
            "import json,pathlib,sys,zipfile\nfrom xml.etree import ElementTree as ET\n"
            "p=json.load(sys.stdin)\n"
            "with zipfile.ZipFile(pathlib.Path(p['workspace_dir'])/'input.docx') as z:\n"
            " text=''.join(ET.fromstring(z.read('word/document.xml')).itertext())\n"
            "print(json.dumps({'text':text},ensure_ascii=False))\n"
        )
        item = synthetic_input()
        answer = json.dumps({
            key: item[key] for key in ("case_id", "document_sha256", "text_sha256")
        } | {
            "schema_version": "sc35-span-review-v1", "taxonomy_sha256": "c" * 64,
            "extraction_version": "sc35-corrupted-text-v1", "offset_unit": "unicode_codepoint",
            "status": "ready", "findings": [], "unresolved": [],
        })
        gateway = Gateway(scenario, workspace / "input.docx", answer)
        server = ThreadingHTTPServer(("127.0.0.1", 0), gateway.handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            failed = False
            result = None
            with open(os.devnull, "w") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    runtime.prepare_home(home)
                    url = f"http://127.0.0.1:{server.server_port}/runs/synthetic/{SECRET}/v1"
                    state = runtime.install(url, 15)
                    from run_agent import AIAgent
                    from tools.registry import registry

                    task = {
                        "input": json.dumps(item), "model": runtime.MODEL,
                        "model_base_url": url, "model_request_timeout": 15,
                        "prepare_timeout": 30, "parameters": {"toolsets": []},
                    }
                    try:
                        result = solve(task, candidate, assets, workspace, AIAgent, registry)
                        state.check()
                    except (ValueError, RuntimeError):
                        failed = True
                    settings = effective_settings()
            messages = [message for request in gateway.requests for message in request["messages"]]
            tool_results = [message["content"] for message in messages if message["role"] == "tool"]
            success = scenario == "success"
            checks = {
                "expected_result": result == answer and not failed if success else failed,
                "request_count": len(gateway.requests) == (2 if success else 1),
                "document_tool_result": (
                    any(FACT in text for text in tool_results) if success else True
                ),
                "no_base64_context": item["docx_base64"] not in json.dumps(messages),
                "fixed_settings": all(settings.values()) and not gateway.invalid_requests,
                "no_secret_files": all(
                    SECRET.encode() not in path.read_bytes()
                    for path in root.rglob("*") if path.is_file()
                ),
            }
            return {"scenario": scenario, "passed": all(checks.values()), "checks": checks,
                    "model_requests": len(gateway.requests)}
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("success", "failure"))
    args = parser.parse_args()
    if args.scenario:
        report = check(args.scenario)
        print(json.dumps(report))
        return 0 if report["passed"] else 1
    reports = []
    for scenario in ("success", "failure"):
        process = subprocess.run(
            [sys.executable, "-m", "cases.sc35.acceptance.hermes_check", "--scenario", scenario],
            capture_output=True, text=True, timeout=180, check=False,
        )
        if process.returncode:
            raise RuntimeError(f"Synthetic SC35 Hermes {scenario} check failed: {process.stdout}")
        reports.append(json.loads(process.stdout))
    print(json.dumps({"passed": True, "scenarios": reports}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
