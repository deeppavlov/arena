"""Shared structural preparation and retrieval; strategy settings are unchanged."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from . import api as base
from . import common
from .chunking import parse_document


def ranges_for(db, chunk, origin, start, end, config, expand):
    node_id, heading_path = db.execute(
        "SELECT node,heading_path FROM chunk_nodes WHERE chunk=?", (chunk,)
    ).fetchone()
    rows = db.execute(
        "SELECT id,parent,kind,label,start,end,header_end FROM nodes WHERE origin=?", (origin,)
    )
    nodes = {
        r[0]: dict(
            zip(("id", "parent", "kind", "label", "start", "end", "header_end"), r, strict=True)
        )
        for r in rows
    }
    node = nodes[node_id]
    extra = {
        "node_id": node_id,
        "node_kind": node["kind"],
        "heading_path": json.loads(heading_path),
        "role": "context",
    }
    seed = dict(start=start, end=end, section=node["label"], **extra)
    if not expand:
        return [seed]
    limit = config["parent_chars"]
    if node["kind"] == "table":
        if node["end"] - node["start"] <= limit:
            return [dict(seed, start=node["start"], end=node["end"])]
        if start >= node["header_end"]:
            return [
                dict(seed, start=node["start"], end=node["header_end"], role="table_header"),
                seed,
            ]
        return [seed]
    parent = nodes.get(node["parent"])
    if parent and parent["kind"] == "clause" and parent["end"] - parent["start"] <= limit:
        return [dict(seed, start=parent["start"], end=parent["end"], section=parent["label"])]
    if node["end"] - node["start"] <= limit:
        left, right = node["start"], node["end"]
        if node["kind"] == "clause" and config["neighbor_clauses"]:
            siblings = sorted(
                (
                    n
                    for n in nodes.values()
                    if n["parent"] == node["parent"] and n["kind"] == "clause"
                ),
                key=lambda n: n["start"],
            )
            index = next(i for i, n in enumerate(siblings) if n["id"] == node_id)
            distance = config["neighbor_clauses"]
            for neighbor in siblings[max(0, index - distance) : index + distance + 1]:
                lo, hi = min(left, neighbor["start"]), max(right, neighbor["end"])
                if hi - lo <= limit:
                    left, right = lo, hi
        return [dict(seed, start=left, end=right)]
    return [seed]


class StructuralBackend:
    def __init__(self, name, package):
        self.name, self.package = name, package

    def defaults(self):
        value = common.settings(self.name)
        value["structure"] = {"parent_chars": 3600, "neighbor_clauses": 1, "heading_weight": 0.15}
        return value

    def validate(self, value):
        common.validate_settings(value, self.name)
        config = value["structure"]
        if type(config["parent_chars"]) is not int or config["parent_chars"] < 128:
            raise ValueError("parent_chars must be at least 128")
        if type(config["neighbor_clauses"]) is not int or not 0 <= config["neighbor_clauses"] <= 2:
            raise ValueError("neighbor_clauses must be 0..2")
        if (
            type(config["heading_weight"]) not in (int, float)
            or not 0 <= config["heading_weight"] <= 1
        ):
            raise ValueError("heading_weight must be in 0..1")

    def build(self, kb: Path, output: Path, value):
        self.validate(value)
        started = time.monotonic()
        kb, output = kb.resolve(), output.resolve()
        if (kb / "norm").is_dir():
            kb = kb / "norm"
        if output == kb or kb in output.parents or output in kb.parents:
            raise ValueError("index must be outside the corpus")
        if output.exists() and any(output.iterdir()):
            raise ValueError("build directory must be empty; use a new version")
        paths = sorted(kb.glob("*.md"))
        if not paths or any(not base.ORIGIN.fullmatch(p.stem) or p.is_symlink() for p in paths):
            raise ValueError("corpus must contain regular 64-hex named Markdown documents")
        output.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(output / "index.sqlite3")
        source_hashes, profiles, count = [], Counter(), 0
        try:
            db.executescript("""
            CREATE TABLE documents(origin TEXT PRIMARY KEY, sha256 TEXT, body TEXT);
            CREATE TABLE chunks(id INTEGER PRIMARY KEY, origin TEXT, seq INTEGER, section TEXT,
              start INTEGER, end INTEGER, line_start INTEGER, line_end INTEGER,
              body TEXT, length INTEGER);
            CREATE INDEX chunk_document ON chunks(origin,seq);
            CREATE TABLE postings(term TEXT, chunk INTEGER, tf INTEGER, PRIMARY KEY(term,chunk));
            CREATE TABLE vocabulary(term TEXT PRIMARY KEY, df INTEGER);
            CREATE TABLE document_meta(origin TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE nodes(origin TEXT, id INTEGER, parent INTEGER, kind TEXT, label TEXT,
              start INTEGER, end INTEGER, header_end INTEGER, PRIMARY KEY(origin,id));
            CREATE TABLE chunk_nodes(chunk INTEGER PRIMARY KEY, node INTEGER, heading_path TEXT);
            """)
            for path in paths:
                raw = path.read_bytes()
                text = raw.decode("utf-8-sig")
                parsed = parse_document(text, value["lexical"]["chunk_chars"])
                origin, hashed = path.stem, base.digest(raw)
                source_hashes.append((origin, hashed))
                profiles[parsed["metadata"]["doc_type"]] += 1
                db.execute("INSERT INTO documents VALUES (?,?,?)", (origin, hashed, text))
                db.execute(
                    "INSERT INTO document_meta VALUES (?,?)",
                    (origin, json.dumps(parsed["metadata"], ensure_ascii=False)),
                )
                db.executemany(
                    "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                    [
                        (
                            origin,
                            n["id"],
                            n["parent"],
                            n["kind"],
                            n["label"],
                            n["start"],
                            n["end"],
                            n["header_end"],
                        )
                        for n in parsed["nodes"]
                    ],
                )
                for seq, chunk in enumerate(parsed["chunks"]):
                    terms = Counter(base.terms(chunk["text"]))
                    db.execute(
                        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            count,
                            origin,
                            seq,
                            chunk["section"],
                            chunk["start"],
                            chunk["end"],
                            chunk["line_start"],
                            chunk["line_end"],
                            chunk["text"],
                            sum(terms.values()),
                        ),
                    )
                    db.executemany(
                        "INSERT INTO postings VALUES (?,?,?)",
                        [(term, count, tf) for term, tf in terms.items()],
                    )
                    db.execute(
                        "INSERT INTO chunk_nodes VALUES (?,?,?)",
                        (count, chunk["node_id"], json.dumps(chunk["heading_path"])),
                    )
                    count += 1
                db.commit()
            if not count:
                raise ValueError("corpus contains no searchable text")
            db.execute("INSERT INTO vocabulary SELECT term,COUNT(*) FROM postings GROUP BY term")
            average = db.execute("SELECT AVG(length) FROM chunks").fetchone()[0]
            db.commit()
        finally:
            db.close()
        manifest = {
            "schema_version": 1,
            "config": value["lexical"],
            "documents": len(paths),
            "chunks": count,
            "corpus_sha256": base.digest(json.dumps(source_hashes).encode()),
            "sources": source_hashes,
            "dimension": None,
            "mean_length": average,
            "edges": 0,
            "profiles": dict(profiles),
            "parser_version": "source-tree-v1",
            "build_seconds": time.monotonic() - started,
            "python": sys.version.split()[0],
        }
        return common.seal_build(output, manifest, value, self.package)

    def retrieve(self, output: Path, question: str, *, expand=True, heading_bonus=True):
        started = time.monotonic()
        question = base.extract_question(question)
        manifest = common.load_build(output, self.name)
        value = manifest["candidate_settings"]
        self.validate(value)
        db = common.open_index(output)
        try:
            ranked = base.bm25(db, question, manifest, value["lexical"]["pool_k"])
            if heading_bonus:
                query_terms = set(base.terms(question))
                boosted = []
                for chunk, score in ranked:
                    path = db.execute(
                        "SELECT heading_path FROM chunk_nodes WHERE chunk=?", (chunk,)
                    ).fetchone()[0]
                    overlap = len(query_terms & set(base.terms(" ".join(json.loads(path)))))
                    fraction = overlap / max(1, len(query_terms))
                    boosted.append(
                        (chunk, score * (1 + value["structure"]["heading_weight"] * fraction))
                    )
                ranked = sorted(boosted, key=lambda item: (-item[1], item[0]))

            def provider(chunk, origin, start, end):
                return ranges_for(db, chunk, origin, start, end, value["structure"], expand)

            hits, used = common.select_hits(db, ranked, manifest, provider)
            for hit in hits:
                hit["metadata"] = json.loads(
                    db.execute(
                        "SELECT value FROM document_meta WHERE origin=?", (hit["origin_id"],)
                    ).fetchone()[0]
                )
        finally:
            db.close()
        return {
            "method": self.name,
            "corpus_sha256": manifest["corpus_sha256"],
            "status": "retrieved" if hits else "no_candidates",
            "hits": hits,
            "links": [],
            "query_sha256": base.digest(question.encode()),
            "lexical_candidates": len(ranked),
            "dense_candidates": 0,
            "rerank": False,
            "structure_expansion": bool(expand),
            "heading_bonus": bool(heading_bonus),
            "context_chars_used": used,
            "model_calls_retrieval": 0,
            "seconds": time.monotonic() - started,
            "provenance": common.provenance(output, self.package),
        }
