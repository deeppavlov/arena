"""Model transport and unchanged dense/reranker encoding."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import Any


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("model endpoint redirects are not allowed")


def post_json(url: str, payload: dict[str, Any], key_env: str, timeout: int) -> dict[str, Any]:
    """Internal endpoints plus the explicitly approved laboratory chat route."""
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model endpoint must be an http(s) URL without credentials")
    laboratory = (parsed.scheme, parsed.hostname.lower(), parsed.port, parsed.path) == (
        "http",
        "lnsigo.mipt.ru",
        4000,
        "/v1/chat/completions",
    )
    addresses = socket.getaddrinfo(
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not (
            laboratory
            or ip.is_loopback
            or ip in ipaddress.ip_network("10.0.0.0/8")
            or ip in ipaddress.ip_network("172.16.0.0/12")
            or ip in ipaddress.ip_network("192.168.0.0/16")
        ):
            raise ValueError("public model endpoints are disabled for this closed corpus")
    headers = {"Content-Type": "application/json"}
    if os.environ.get(key_env):
        headers["Authorization"] = "Bearer " + os.environ[key_env]
    request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    for attempt in range(3):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("model request exhausted its timeout budget")
            with opener.open(request, timeout=remaining) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as error:
            # Retry only an explicit cold-start rejection, never a completed answer.
            retry = error.headers.get("Retry-After") if error.headers else None
            delay = None
            if error.code == 503 and retry is not None:
                try:
                    delay = max(0.0, float(retry))
                except ValueError:
                    try:
                        delay = max(
                            0.0, (parsedate_to_datetime(retry) - datetime.now(UTC)).total_seconds()
                        )
                    except (ValueError, TypeError, OverflowError):
                        pass
            if attempt < 2 and delay is not None and delay < deadline - time.monotonic():
                error.close()
                time.sleep(delay)
                continue
            # Do not leak response text, credentials or document bodies into logs.
            raise ValueError(f"model endpoint returned HTTP {error.code}") from None
    else:
        raise ValueError("model request exhausted its retry budget")
    if not isinstance(data, dict):
        raise ValueError("model endpoint returned a non-object")
    return data


class Encoder:
    def __init__(self, config: dict[str, Any], *, device=None):
        self.config = config
        self.model: Any = None
        self.url = os.environ.get("SC71_EMBED_URL", "")
        if config["embedding_backend"] == "http" and not self.url:
            raise ValueError("SC71_EMBED_URL must name an internal /v1/embeddings endpoint")
        if config["embedding_backend"] == "local":
            path = os.environ.get("SC71_LOCAL_EMBED_MODEL", "")
            if not Path(path).is_dir() or not path:
                raise ValueError("SC71_LOCAL_EMBED_MODEL must be a preloaded local model directory")
            self.model = import_module("sentence_transformers").SentenceTransformer(
                path,
                device=device or os.environ.get("SC71_DEVICE", "cpu"),
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model.max_seq_length = 2048

    def encode(self, texts: list[str]):
        import numpy as np

        if self.model is not None:
            rows = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            data = post_json(
                self.url,
                {"model": self.config["embedding_model"], "input": texts},
                "SC71_EMBED_KEY",
                self.config["timeout_seconds"],
            )
            items = data.get("data", [])
            if len(items) != len(texts) or {i["index"] for i in items} != set(range(len(texts))):
                raise ValueError("embedding response indices/count mismatch")
            if "model" in data and data["model"] != self.config["embedding_model"]:
                raise ValueError("embedding response model mismatch")
            rows = [i["embedding"] for i in sorted(items, key=lambda i: i["index"])]
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(texts) or not np.isfinite(matrix).all():
            raise ValueError("invalid embedding matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("zero embedding")
        return matrix / norms


def model_identity(config, revision=None):
    """Identify the configured local model or remote revision."""
    identity = {
        "backend": config["embedding_backend"],
        "model": config["embedding_model"],
        "max_chars": config["embedding_max_chars"],
        "max_seq_length": 2048,
        "normalize": True,
        "dtype": "float32",
        "prefix": "header400-newline-body-v1",
    }
    if identity["backend"] == "local":
        root = Path(os.environ.get("SC71_LOCAL_EMBED_MODEL", ""))
        if not os.environ.get("SC71_LOCAL_EMBED_MODEL") or not root.is_dir():
            raise ValueError("SC71_LOCAL_EMBED_MODEL must name preloaded weights")
        if not any(
            p.is_file() and ".cache" not in p.relative_to(root).parts
            for p in root.rglob("*")
        ):
            raise ValueError("embedding weights directory is empty")
        identity["model_path"] = str(root.resolve())
    else:
        revision = revision or os.environ.get("SC71_EMBED_REVISION")
        if not revision:
            raise ValueError("HTTP embedding cache requires --embedding-revision from the operator")
        identity["revision"] = revision
    return identity


def validate_devices(config, devices):
    if devices is None:
        return None
    if config["embedding_backend"] != "local":
        raise ValueError("explicit devices require the local embedding backend")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a nonempty list without duplicates")
    if any(not re.fullmatch(r"cuda:\d+", d) for d in devices):
        raise ValueError("use logical CUDA devices, e.g. --devices cuda:0 cuda:1")
    return list(devices)


def prefix_metadata(index, config, completed, dimension):
    """Describe a contiguous prefix with its sources and encoder settings."""
    index = Path(index)
    manifest = json.loads((index / "build.json").read_text(encoding="utf-8"))
    return {
        "schema": "sc71-embedding-prefix-v1",
        "chunks": manifest["chunks"],
        "sources": manifest["sources"],
        "embedding_identity": model_identity(config),
        "completed_rows": completed,
        "dimension": dimension,
        "batch_per_device": config["embedding_batch"],
    }


def read_prefix(source, expected):
    """Read complete rows, checking dimensions and normalized vector values."""
    import numpy as np

    source = Path(source)
    metadata = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    for key in ("schema", "chunks", "sources", "embedding_identity"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"checkpoint {key} differs from the current inputs")
    count, dimension = metadata.get("completed_rows"), metadata.get("dimension")
    if type(count) is not int or not 0 <= count <= expected["chunks"]:
        raise ValueError("invalid checkpoint row count")
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("invalid checkpoint dimension")
    path = source / "vectors.f32"
    if path.is_symlink() or not path.is_file() or path.stat().st_size < count * dimension * 4:
        raise ValueError("checkpoint vectors are missing or truncated")
    with path.open("rb") as stream:
        remaining = count
        while remaining:
            rows = min(remaining, 4096)
            raw = stream.read(rows * dimension * 4)
            vectors = np.frombuffer(raw, dtype="<f4").reshape(rows, dimension)
            if not np.isfinite(vectors).all() or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1, atol=1e-4
            ):
                raise ValueError("checkpoint contains invalid/non-normalized vectors")
            remaining -= rows
    return metadata


_worker_encoder = None


def _init_embedding_worker(config, device_queue):
    """One model per selected GPU. No model or data is sent outside this host."""
    global _worker_encoder
    _worker_encoder = Encoder(config, device=device_queue.get())


def _encode_worker(texts):
    if _worker_encoder is None:
        raise RuntimeError("embedding worker was not initialized")
    return _worker_encoder.encode(texts)


def prepare_matrix(index, output, config, *, resume_from=None, devices=None, progress=False):
    """Stream ordered batches; retain completed rows and checkpoint every block."""
    import numpy as np

    devices = validate_devices(config, devices)
    output = Path(output)
    if resume_from is not None and Path(resume_from).resolve() == output.resolve():
        raise ValueError("checkpoint source and destination must be different")
    expected = prefix_metadata(index, config, 0, 1)
    prefix = read_prefix(resume_from, expected) if resume_from else None
    count = prefix["completed_rows"] if prefix else 0
    dimension = prefix["dimension"] if prefix else None
    resumed = count
    started = time.monotonic()
    executor = device_queue = None
    db = sqlite3.connect((index / "index.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        extent = db.execute("SELECT MIN(id),MAX(id) FROM chunks").fetchone()
        if chunks != expected["chunks"] or (chunks and extent != (0, chunks - 1)):
            raise ValueError("index rows are not the expected contiguous chunk IDs")
        batch = config["embedding_batch"]
        block = batch * max(1, len(devices or [])) * 8
        with (output / "vectors.f32").open("wb") as stream:
            if prefix is not None and resume_from is not None:
                with (Path(resume_from) / "vectors.f32").open("rb") as source:
                    remaining = count * int(prefix["dimension"]) * 4
                    while remaining:
                        raw = source.read(min(remaining, 1024 * 1024))
                        if not raw:
                            raise ValueError("source checkpoint changed while copying")
                        stream.write(raw)
                        remaining -= len(raw)

            def commit():
                from .storage import write_json

                stream.flush()
                os.fsync(stream.fileno())
                metadata = dict(
                    expected,
                    completed_rows=count,
                    dimension=dimension,
                )
                temporary = output / ".checkpoint.next"
                write_json(temporary, metadata)
                os.replace(temporary, output / "checkpoint.json")
                if progress:
                    elapsed = time.monotonic() - started
                    rate = (count - resumed) / elapsed if count > resumed else None
                    print(
                        json.dumps(
                            {
                                "phase": "embeddings", "done": count, "total": chunks,
                                "percent": round(100 * count / chunks, 2) if chunks else 0,
                                "resumed_rows": resumed,
                                "rows_per_second": round(rate, 2) if rate else None,
                                "eta_minutes": round((chunks - count) / rate / 60, 1)
                                if rate else None,
                            }
                        ),
                        file=sys.stderr, flush=True,
                    )

            if prefix:
                commit()  # Preserve imported work even if model loading fails next.
            if count < chunks:
                encoder = None
                if devices:
                    context = get_context("spawn")
                    device_queue = context.Queue()
                    for device in devices:
                        device_queue.put(device)
                    executor = ProcessPoolExecutor(
                        max_workers=len(devices), mp_context=context,
                        initializer=_init_embedding_worker, initargs=(config, device_queue),
                    )
                else:
                    encoder = Encoder(config)
                for start in range(count, chunks, block):
                    rows = list(db.execute(
                        "SELECT c.body, substr(d.body,1,400) FROM chunks c JOIN documents d"
                        " ON c.origin=d.origin WHERE c.id>=? AND c.id<? ORDER BY c.id",
                        (start, min(start + block, chunks)),
                    ))
                    texts = [
                        (header[:400] + "\n" + body)[: config["embedding_max_chars"]]
                        for body, header in rows
                    ]
                    batches = [texts[i:i + batch] for i in range(0, len(texts), batch)]
                    if executor:
                        pending = [executor.submit(_encode_worker, texts) for texts in batches]
                        # Future completion order never changes matrix row order.
                        vectors_list = [future.result() for future in pending]
                    else:
                        assert encoder is not None
                        vectors_list = [encoder.encode(texts) for texts in batches]
                    for vectors, texts in zip(vectors_list, batches, strict=True):
                        vectors = np.asarray(vectors, dtype=np.float32)
                        if (vectors.ndim != 2 or len(vectors) != len(texts)
                                or not np.isfinite(vectors).all()):
                            raise ValueError("invalid embedding block")
                        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                            raise ValueError("non-normalized embedding block")
                        if dimension is None:
                            dimension = vectors.shape[1]
                        if vectors.shape[1] != dimension:
                            raise ValueError("embedding dimension changed within build")
                        raw = np.asarray(vectors, dtype="<f4").tobytes()
                        stream.write(raw)
                        count += len(vectors)
                    commit()
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        if device_queue is not None:
            device_queue.close()
            device_queue.join_thread()
        db.close()
    return {
        "chunks": count, "dimension": dimension,
        "execution": {
            "devices": devices or [os.environ.get("SC71_DEVICE", "cpu")],
            "batch_per_device": config["embedding_batch"],
            "resumed_rows": resumed,
            "source_provenance": prefix.get("legacy_import") if prefix else None,
            "seconds": time.monotonic() - started,
        },
    }


def rerank(query: str, candidates: list[tuple[int, float]], db: sqlite3.Connection, config):
    if not candidates:
        return candidates
    texts = [
        db.execute("SELECT body FROM chunks WHERE id=?", (i,)).fetchone()[0] for i, _ in candidates
    ]
    url = os.environ.get("SC71_RERANK_URL")
    if url:
        data = post_json(
            url,
            {
                "model": config["reranker_model"],
                "query": query,
                "documents": texts,
                "top_n": len(texts),
            },
            "SC71_RERANK_KEY",
            config["timeout_seconds"],
        )
        results = data.get("results", [])
        if len(results) != len(texts) or {r["index"] for r in results} != set(range(len(texts))):
            raise ValueError("rerank response indices mismatch")
        ranked = [(candidates[r["index"]][0], float(r["relevance_score"])) for r in results]
    else:
        path = os.environ.get("SC71_LOCAL_RERANK_MODEL", "")
        if not path or not Path(path).is_dir():
            raise ValueError("rerank requires SC71_RERANK_URL or SC71_LOCAL_RERANK_MODEL")
        model = import_module("sentence_transformers").CrossEncoder(
            path,
            device=os.environ.get("SC71_DEVICE", "cpu"),
            local_files_only=True,
            trust_remote_code=False,
            max_length=1024,
        )
        ranked = [
            (candidate[0], float(score))
            for candidate, score in zip(
                candidates, model.predict([(query, t) for t in texts]), strict=True
            )
        ]
    if any(not math.isfinite(score) for _, score in ranked):
        raise ValueError("nonfinite reranker score")
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
