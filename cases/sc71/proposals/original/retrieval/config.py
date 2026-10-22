"""Unchanged candidate defaults and lexical/citation patterns."""

from __future__ import annotations

import re
from typing import Any

ORIGIN = re.compile(r"^[a-f0-9]{64}$")
WORDS = re.compile(r"[a-zа-яё]+|\d+(?:[./-]\d+)*(?:-[a-zа-яё0-9]+)?", re.I)
BOUNDARY = re.compile(r"^\s*(?:#{1,6}\s+|(?P<section>\d+(?:\.\d+){0,5})[.)]?\s+)")
DESIGNATION = re.compile(
    r"(?P<kind>приказ|распоряжение|положение)\b[^\n]{0,100}?"
    r"(?:от\s*(?P<date>\d{2}\.\d{2}\.\d{4})\s*№\s*(?P<num>[\w/-]+)"
    r"|№\s*(?P<num2>[\w/-]+)\s*от\s*(?P<date2>\d{2}\.\d{2}\.\d{4}))",
    re.I,
)
SOURCE_LINK = re.compile(r"/kb/norm/([a-f0-9]{64})\.md")
STOP = set(
    "какой какая какие какое кто где когда как ли и в во на по из для это что чем"
    " или о об с со к у а но мы вы мне нужно можно должен должна должны мфти".split()
)
GRAPH_QUERY = re.compile(
    r"поряд|регламент|этап|соглас|измен|отмен|редакци|коллизи|противор|связ", re.I
)


def defaults(method: str) -> dict[str, Any]:
    if method not in {"bm25_sections", "hybrid_bge", "linked_hybrid"}:
        raise ValueError("unknown candidate")
    return {
        "method": method,
        "chunk_chars": 1800,
        "overlap_lines": 2,
        "top_k": 8,
        "pool_k": 48,
        "max_per_document": 3,
        "context_chars": 16000,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "rrf_k": 60,
        "embedding_backend": "http",
        "embedding_model": "BAAI/bge-m3",
        "embedding_batch": 16,
        "embedding_max_chars": 2400,
        "dense_min_cosine": 0.0,
        "rerank": False,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "graph_depth": 1,
        "graph_documents": 6,
        "timeout_seconds": 60,
    }


def validate_config(c: dict[str, Any]) -> None:
    if c["method"] not in {"bm25_sections", "hybrid_bge", "linked_hybrid"}:
        raise ValueError("unknown method")
    for key in (
        "chunk_chars",
        "top_k",
        "pool_k",
        "max_per_document",
        "context_chars",
        "embedding_batch",
        "embedding_max_chars",
        "timeout_seconds",
        "graph_documents",
    ):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["chunk_chars"] < 128 or c["embedding_batch"] > 256:
        raise ValueError("chunk/batch limits are invalid")
    if c["overlap_lines"] < 0 or c["graph_depth"] not in (1, 2):
        raise ValueError("overlap/depth is invalid")
    if c["bm25_k1"] <= 0 or not 0 <= c["bm25_b"] <= 1 or c["rrf_k"] <= 0:
        raise ValueError("BM25/RRF parameters are invalid")
    if c["method"] != "bm25_sections" and c["embedding_backend"] not in {"http", "local"}:
        raise ValueError("hybrid requires a real http or local embedding backend")
    if c["method"] != "bm25_sections" and c["embedding_max_chars"] < c["chunk_chars"] + 401:
        raise ValueError("embedding input limit must cover header and complete chunk")
