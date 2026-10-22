"""Arena context adapter, contract commit 4858f0bf02c3766b84d8dd4bce7810390b9028bc.

Corpus preparation is deliberately absent. Gateway capabilities stay in memory.
"""

import json
import math
import os
import sys
import urllib.request
from pathlib import Path

from retrieval import api, embeddings, search, storage


def asset_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("asset path must be relative")
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError("asset path escapes assets_dir")
    return resolved


def context(task, selection=None):
    if not isinstance(task.get("input"), str):
        raise ValueError("input must be text")
    parameters = dict(selection or {})
    parameters.update(task.get("parameters", {}))
    name = parameters["candidate"]
    root = Path(task["assets_dir"]).resolve()
    index = asset_path(root, parameters.get("index", "indexes/" + name))
    manifest = json.loads((index / "build.json").read_text(encoding="utf-8"))
    actual = manifest.get("candidate", manifest["config"]["method"])
    if actual != name:
        raise ValueError("selected candidate and prepared index differ")
    candidate = storage.load_candidate(name)
    keys = ("SC71_CHAT_URL", "SC71_CHAT_MODEL", "SC71_DEVICE", "SC71_LOCAL_EMBED_MODEL")
    previous = {key: os.environ.get(key) for key in keys}
    transport = api.post_json

    def gateway(url, payload, key_env, timeout):
        if key_env != "SC71_CHAT_KEY":
            raise ValueError("this profile supports only the fixed Arena chat gateway")
        base = task["model_base_url"]
        limit = task["model_request_timeout"]
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("invalid model timeout")
        # No request URL, payload, capability, or upstream response body is logged.
        try:
            request = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                json.dumps({
                    "model": task["model"],
                    "messages": payload["messages"],
                }).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), embeddings.NoRedirect()
            )
            with opener.open(request, timeout=limit) as response:
                return json.load(response)
        except Exception:
            raise ValueError(
                "Arena gateway request failed; inspect the operator's retained diagnostics"
            ) from None

    try:
        os.environ["SC71_DEVICE"] = "cpu"
        # This placeholder has no secret. Only gateway() uses the real capability from stdin.
        os.environ["SC71_CHAT_URL"] = "http://127.0.0.1/v1/chat/completions"
        os.environ["SC71_CHAT_MODEL"] = task.get("model", "")
        if name in {"hybrid_bge", "linked_hybrid"}:
            if manifest["config"]["embedding_backend"] != "local":
                raise ValueError("HTTP embeddings require a separately approved Arena profile")
            os.environ["SC71_LOCAL_EMBED_MODEL"] = str(
                asset_path(root, parameters["embedding_model"])
            )
        if manifest["config"]["rerank"]:
            raise ValueError("reranking requires a separately approved Arena profile")
        api.post_json = gateway
        result = candidate.retrieve(index, task["input"])
        return {
            "context": json.dumps(search.answer_context(task["input"], result), ensure_ascii=False)
        }
    finally:
        api.post_json = transport
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    try:
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("input too large")
        task = json.loads(raw)
        root = next(p for p in Path(__file__).resolve().parents if (p / "retrieval").is_dir())
        selection = root / "selection.json"
        defaults = json.loads(selection.read_text(encoding="utf-8")) if selection.exists() else {}
        sys.stdout.buffer.write(
            json.dumps(context(task, defaults), ensure_ascii=False).encode("utf-8") + b"\n"
        )
    except Exception as error:
        # Errors can contain an upstream URL; emit only the exception type at this boundary.
        print(
            json.dumps({"stage": "arena-prepare", "error_type": type(error).__name__}),
            file=sys.stderr,
        )
        raise SystemExit(2) from None
