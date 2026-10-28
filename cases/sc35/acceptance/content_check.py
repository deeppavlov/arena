import base64
import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
SECRET = "synthetic-content-capability"


def synthetic_input():
    names = [
        "Объект испытаний", "Цель испытаний", "Требования\u00a0к\u00a0программе",
        "Требования\u00a0к\u00a0программной документации",
        "Средства\u00a0и\u00a0порядок испытаний", "Методы испытаний",
    ]
    texts = [text for name in names for text in (name, "Синтетическое описание 😀.")]
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    document = ET.Element(namespace + "document")
    body = ET.SubElement(document, namespace + "body")
    for text in texts:
        paragraph = ET.SubElement(body, namespace + "p")
        run = ET.SubElement(paragraph, namespace + "r")
        ET.SubElement(run, namespace + "t").text = text
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", ET.tostring(document))
    data = output.getvalue()
    return {
        "case_id": "synthetic-content",
        "docx_base64": base64.b64encode(data).decode(),
        "document_sha256": hashlib.sha256(data).hexdigest(),
        "text_sha256": hashlib.sha256(("\n".join(texts) + "\n").encode()).hexdigest(),
    }


def check(scenario):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if scenario == "failure":
                status, response = 500, {"error": {"message": "Synthetic content failure"}}
            else:
                data = json.loads(payload["messages"][1]["content"])
                answer = {
                    "assessments": [{
                        "rule_id": rule["rule_id"], "status": "pass",
                        "reason": "Synthetic transport fixture, not a quality judgment.",
                    } for rule in data["requirements"]],
                    "findings": [],
                }
                status, response = 200, {
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps(answer, ensure_ascii=False),
                    }}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            encoded = json.dumps(response, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        item = synthetic_input()
        task = {
            "input": json.dumps(item, ensure_ascii=False), "uses_model": True,
            "model": MODEL, "model_request_timeout": 15, "prepare_timeout": 30,
            "model_base_url": f"http://127.0.0.1:{server.server_port}/runs/synthetic/{SECRET}/v1",
        }
        process = subprocess.run(
            [sys.executable, "/opt/arena/cases/sc35/harness/content.py"],
            input=json.dumps(task, ensure_ascii=False), text=True, capture_output=True,
            timeout=120, check=False,
        )
        checks = {
            "one_model_request": len(requests) == 1,
            "fixed_model": bool(requests) and all(p["model"] == MODEL for p in requests),
            "no_secret_output": SECRET not in process.stdout + process.stderr,
        }
        if scenario == "success":
            response = json.loads(process.stdout) if process.returncode == 0 else {}
            review = json.loads(response.get("answer", "{}"))
            checks.update({
                "entrypoint_success": process.returncode == 0,
                "one_runner_object": set(response) == {"answer"},
                "review_ready": review.get("schema_version") == "sc35-span-review-v1"
                and review.get("status") == "ready",
                "four_stages": review.get("stages") == [
                    "graph", "format", "typography", "semantic-llm"
                ],
                "forty_semantic_rules": len(review.get("semantic_assessments", [])) == 40,
                "all_rule_statuses": len(review.get("rule_assessments", [])) == 113,
                "http_model_called": review.get("llm_called") is True,
                "identity_preserved": all(review.get(key) == item[key] for key in (
                    "case_id", "document_sha256", "text_sha256"
                )),
            })
        else:
            checks.update({"entrypoint_failure": process.returncode != 0,
                           "no_answer_after_failure": not process.stdout.strip()})
        return {"scenario": scenario, "passed": all(checks.values()), "checks": checks,
                "model_requests": len(requests), "returncode": process.returncode}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    reports = [check(scenario) for scenario in ("success", "failure")]
    passed = all(report["passed"] for report in reports)
    print(json.dumps({"passed": passed, "scenarios": reports}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
