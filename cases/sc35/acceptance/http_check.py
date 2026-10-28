import argparse
import base64
import contextlib
import hashlib
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
CAMPAIGN = "sc35-synthetic-http-20261028"
DATASET = "sc35-synthetic-http-v1"
CONDITIONS = "sc35-synthetic-http-v1"
CANDIDATE = "synthetic-hermes-http"
MEASUREMENT = "synthetic-http-1"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixtures(metric):
    taxonomy = json.loads((metric / "taxonomy.json").read_text(encoding="utf-8"))
    rule = next(key for key, category in taxonomy["rule_categories"].items()
                if category == "sections_content")
    text = "Синтетический документ для проверки транспорта.\n"
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    document = ElementTree.Element(namespace + "document")
    body = ElementTree.SubElement(document, namespace + "body")
    paragraph = ElementTree.SubElement(body, namespace + "p")
    run = ElementTree.SubElement(paragraph, namespace + "r")
    ElementTree.SubElement(run, namespace + "t").text = text.rstrip("\n")
    stream = io.BytesIO()
    with ZipFile(stream, "w") as package:
        package.writestr("word/document.xml", ElementTree.tostring(document))
    identity = {
        "case_id": "synthetic-http",
        "document_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "extraction_version": "sc35-corrupted-text-v1", "offset_unit": "unicode_codepoint",
    }
    finding = {
        "id": "F1", "rule_id": rule, "category": "sections_content",
        "message": "Synthetic transport fixture", "spans": [{"start": 0, "end": 13}],
        "anchor_kind": "object",
    }
    review = {
        **identity, "schema_version": "sc35-span-review-v1", "status": "ready",
        "findings": [finding], "unresolved": [],
    }
    reference = {
        **identity, "schema_version": "sc35-span-reference-v1", "text_length": len(text),
        "findings": [finding],
        "evaluation_scope": {"rule_ids": [rule], "regions": [{"start": 0, "end": len(text)}]},
    }
    task = {key: identity[key] for key in ("case_id", "document_sha256", "text_sha256")}
    task["docx_base64"] = base64.b64encode(stream.getvalue()).decode("ascii")
    example = {
        "id": identity["case_id"], "input": json.dumps(task, ensure_ascii=False),
        "reference": json.dumps(reference, ensure_ascii=False),
    }
    return example, review


def prepare(directory, metric):
    directory.mkdir(parents=True, exist_ok=False)
    example, review = fixtures(metric)
    save(directory / "answer.json", review)
    save(directory / "config.json", {
        "campaign_id": CAMPAIGN, "dataset_id": DATASET, "conditions_id": CONDITIONS,
        "case_id": "sc35-preparation-http", "synthetic_only": True,
    })
    (directory / "dataset.jsonl").write_text(
        json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    candidate = directory / "candidate"
    candidate.mkdir()
    (candidate / "prompt.md").write_text(
        "Это синтетическая проверка транспорта. Верни один окончательный span-review JSON.\n",
        encoding="utf-8",
    )
    save(candidate / "solution.json", {"harness": "hermes", "parameters": {}})
    return {"directory": str(directory), "synthetic_only": True, "examples": 1}


def mock_server(answer, port):
    review = json.loads(answer.read_text(encoding="utf-8"))
    content = json.dumps(review, ensure_ascii=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            body = json.dumps({"data": [{"id": MODEL, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions" or not request.get("messages"):
                self.send_error(400)
                return
            body = json.dumps({
                "id": "sc35-synthetic", "object": "chat.completion", "created": int(time.time()),
                "model": request["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def train_command(arguments):
    from evaluator.train import main

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if main(arguments):
            raise RuntimeError("Train CLI failed")
    return json.loads(output.getvalue())


def wait_for_result(fetch, *, timeout=300, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout
    while True:
        result = fetch()
        if result["status"] in {"FINISHED", "FAILED"}:
            return result
        if result["status"] != "RUNNING":
            raise ValueError("Unknown train status: " + str(result["status"]))
        if clock() >= deadline:
            raise TimeoutError("Synthetic train check did not finish")
        sleep(1)


def run_check(directory, journal_path, url, token_file):
    from evaluator.search import Journal, Resources

    journal = Journal.create(journal_path, max_solutions=1)
    journal.propose(
        CANDIDATE, directory / "candidate", hypothesis="Synthetic HTTP transport and accounting",
        change="Fixed mock response, no real model or benchmark data", resources=Resources(0, 0),
    )
    access = ["--url", url, "--token-file", str(token_file)]
    submitted = train_command([
        "submit", *access, "--journal", str(journal_path), "--candidate", CANDIDATE,
        "--measurement", MEASUREMENT, "--dataset", DATASET, "--conditions", CONDITIONS,
        "--train-data", str(directory / "dataset.jsonl"),
    ])
    evaluation_id = submitted["evaluation_id"]
    def fetch():
        return train_command(["status", *access, "--evaluation-id", evaluation_id])

    terminal = wait_for_result(fetch)
    deadline = time.monotonic() + 300
    while terminal["status"] == "FINISHED" and "tracking" not in terminal:
        if time.monotonic() >= deadline:
            raise TimeoutError("Synthetic tracking export did not finish")
        time.sleep(1)
        terminal = fetch()
    result_path = directory / "result.json"
    result = train_command([
        "result", *access, "--journal", str(journal_path), "--evaluation-id", evaluation_id,
        "--measurement", MEASUREMENT, "--output", str(result_path),
    ])
    if result["status"] != "FINISHED":
        raise RuntimeError("Synthetic train failed; saved result contains diagnostics")
    metrics = result["metrics"]
    if (metrics["score"] != 100 or metrics["solver_tokens"] != 18
            or metrics["judge_tokens"] != 0 or metrics["solver_unreported_attempts"] != 0
            or not metrics["solver_resources_complete"] or not metrics["judge_resources_complete"]):
        raise ValueError("Synthetic score or resource accounting differs")
    status = journal.status()
    if status["running"] or status["completed_measurements"] != 1:
        raise ValueError("Train result was not completed in journal")
    shared = result.get("tracking", {}).get("shared", {})
    if shared.get("status") != "exported":
        raise RuntimeError("Synthetic tracking not exported; retain and flush its queue")
    receipt = {
        "synthetic_only": True, "campaign_id": CAMPAIGN, "evaluation_id": evaluation_id,
        "run_id": shared["run_id"], "score": metrics["score"],
        "solver_tokens": metrics["solver_tokens"], "judge_tokens": metrics["judge_tokens"],
        "solver_seconds": metrics["solver_seconds"], "judge_seconds": metrics["judge_seconds"],
        "journal_completed_measurements": status["completed_measurements"],
    }
    save(directory / "receipt.json", receipt)
    return receipt


def read_trace(receipt_path, uri):
    from mlflow import MlflowClient

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(receipt["run_id"])
    if run.info.status != "FINISHED" or run.data.tags.get("campaign_id") != CAMPAIGN:
        raise ValueError("Synthetic tracking run differs")
    artifact = Path(client.download_artifacts(receipt["run_id"], "record.json"))
    record = json.loads(artifact.read_text())
    if record["metrics"]["score"] != 100 or len(record["examples"]) != 1:
        raise ValueError("Synthetic result payload differs")
    answer = json.loads(record["examples"][0]["answer"])
    if answer["case_id"] != "synthetic-http":
        raise ValueError("Synthetic answer payload missing")
    traces = client.search_traces(
        locations=[run.info.experiment_id], run_id=receipt["run_id"], include_spans=True,
    )
    spans = [span for trace in traces for span in trace.data.spans]
    model_spans = [span for span in spans if span.span_type == "LLM"]
    if not model_spans or not all(span.inputs and span.outputs for span in model_spans):
        raise ValueError("Synthetic model trace payload missing")
    return {
        **receipt, "experiment_id": run.info.experiment_id,
        "trace_ids": [trace.info.trace_id for trace in traces],
        "payload_verified": True, "uri": uri,
    }


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--directory", type=Path, required=True)
    setup.add_argument("--metric", type=Path, required=True)
    mock = commands.add_parser("mock-upstream")
    mock.add_argument("--answer", type=Path, required=True)
    mock.add_argument("--port", type=int, default=8000)
    check = commands.add_parser("run")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--journal", type=Path, required=True)
    check.add_argument("--url", required=True)
    check.add_argument("--token-file", type=Path, required=True)
    trace = commands.add_parser("read-trace")
    trace.add_argument("--receipt", type=Path, required=True)
    trace.add_argument("--uri", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.directory, args.metric)
    elif args.command == "mock-upstream":
        mock_server(args.answer, args.port)
        return
    elif args.command == "run":
        result = run_check(args.directory, args.journal, args.url, args.token_file)
    else:
        result = read_trace(args.receipt, args.uri)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
