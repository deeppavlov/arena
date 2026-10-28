"""Candidate entrypoint for a registered model-enabled SC-35 arena harness."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_pipeline(here):
    runtime = Path(here) / "runtime/sc35"
    sys.path.insert(0, str(runtime / "span"))
    sys.path.insert(0, str(runtime / "arena"))
    from run_graph import load_module

    return load_module("sc35_candidate_pipeline", runtime / "arena/run_baseline.py")


def solve(payload, here=HERE, client_factory=None):
    if (
        not isinstance(payload, dict)
        or payload.get("operation", "solve") != "solve"
        or payload.get("uses_model") is False
    ):
        raise ValueError("SC-35 hybrid runner requires a model-enabled solve request")
    if not isinstance(payload.get("input"), str) or len(payload["input"]) > 500000:
        raise ValueError("Invalid arena example input")
    item = json.loads(payload["input"])
    required = {"case_id", "docx_base64", "document_sha256", "text_sha256"}
    if not isinstance(item, dict) or set(item) != required:
        raise ValueError("Invalid SC-35 input envelope")
    if not isinstance(item["case_id"], str) or not item["case_id"]:
        raise ValueError("Missing case identity")
    data = base64.b64decode(item["docx_base64"], validate=True)
    pipeline = load_pipeline(here)
    taxonomy, profile = pipeline.load_assets(Path(here) / "assets")
    if client_factory is None:
        from content_review import GatewayClient

        client_factory = GatewayClient
    client = client_factory(payload)
    with tempfile.TemporaryDirectory(prefix="sc35-content-") as directory:
        path = Path(directory) / "input.docx"
        path.write_bytes(data)
        projection = pipeline.Projection(path)
        if (
            projection.source_sha256 != item["document_sha256"]
            or projection.text_sha256 != item["text_sha256"]
        ):
            raise ValueError("Input does not match canonical document identity")
        review = pipeline.review_document(path, taxonomy, profile, item["case_id"], client)
    answer = json.dumps(review, ensure_ascii=False, separators=(",", ":"))
    result = {"answer": answer}
    if len((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")) > 1024 * 1024:
        raise ValueError("Combined answer exceeds arena transport limit")
    return result


if __name__ == "__main__":
    try:
        raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Runner input exceeds transport limit")
        result = solve(json.loads(raw))
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as error:
        # Exception strings may include the scoped model URL or confidential text.
        sys.stderr.write(f"SC35 solve failed ({type(error).__name__}); sensitive details omitted\n")
        raise SystemExit(1) from None
