"""Structural BM25 plus bounded bundles of explicit amendments and their targets."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidates.amendment_bm25 import (
    lexical,
    relations,
)
from retrieval import api as base
from retrieval import common
from retrieval import documents as catalog

NAME = "amendment_bm25"
PACKAGE = Path(__file__).resolve().parent


def defaults():
    value = lexical.defaults()
    value["amendments"] = {
        "seed_hits": 3,
        "max_relations": 4,
        "max_span_chars": 3600,
        "max_reference_chars": 2500,
        "include_mentions": False,
    }
    return value


def validate(value):
    lexical.validate(value)
    cfg = value["amendments"]
    for key in ("seed_hits", "max_relations", "max_span_chars", "max_reference_chars"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("amendment budgets must be positive integers")
    if cfg["seed_hits"] > value["lexical"]["top_k"] or cfg["max_relations"] > 8:
        raise ValueError("amendment budgets exceed the bounded protocol")
    if type(cfg["include_mentions"]) is not bool:
        raise ValueError("include_mentions must be boolean")


def prepare_catalog(kb, output, manifest_path=None):
    summary = catalog.prepare_catalog(kb, output, manifest_path)
    rows = [
        json.loads(line)
        for line in (output / "catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary["relations"] = relations.build_relations(rows, kb, output)
    base.write_json(output / "catalog-summary.json", summary)
    return summary


def build(kb, output, value, manifest_path=None):
    validate(value)
    manifest = lexical.build(kb, output, value)
    return finish_preparation(kb, output, manifest, value, manifest_path)


def finish_preparation(kb, output, manifest, value, manifest_path=None):
    """Attach this strategy metadata to an already prepared structural index."""
    try:
        summary, rows, _, _ = catalog.add_catalog(kb, output, manifest_path)
        if summary["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("corpus changed during index construction")
        db = sqlite3.connect(output / "index.sqlite3")
        try:
            summary["relations"] = relations.build_relations(rows, kb, output, db)
        finally:
            db.close()
        manifest["catalog"] = summary
        base.write_json(output / "catalog-summary.json", summary)
        return common.seal_build(output, manifest, value, PACKAGE)
    except Exception:
        (output / "build.json").unlink(missing_ok=True)
        raise


def source_span(db, origin, start, end, role):
    text = db.execute("SELECT body FROM documents WHERE origin=?", (origin,)).fetchone()[0]
    row = db.execute(
        "SELECT id,section FROM chunks WHERE origin=? AND start<? AND end>? ORDER BY start LIMIT 1",
        (origin, end, start),
    ).fetchone()
    return common.source_hit(
        origin,
        text,
        row[0] if row else None,
        0,
        start,
        end,
        row[1] if row else "основание изменения",
        (start, end),
        {"role": role},
    )


def target_span(db, relation, question, manifest, maximum):
    origin = relation["target_origin"]
    nodes = []
    indirect = relation.get("target_annexes") or relation.get("target_role") == "approval_act"
    labels = [] if indirect else relation["target_clauses"]
    for label in labels:
        nodes.extend(
            db.execute(
                (
                    "SELECT start,end FROM nodes WHERE origin=? AND kind='clause' A"
                    "ND label=? ORDER BY start"
                ),
                (origin, label),
            ).fetchall()
        )
    if len(nodes) == 1 and nodes[0][1] - nodes[0][0] <= maximum:
        return source_span(db, origin, nodes[0][0], nodes[0][1], "amendment_target"), True
    # Never identify a repeated or oversized clause as uniquely located.
    ranked = base.bm25(db, question, manifest, manifest["chunks"])
    wanted = {row[0] for row in db.execute("SELECT id FROM chunks WHERE origin=?", (origin,))}
    chunk = next((identifier for identifier, _ in ranked if identifier in wanted), None)
    if chunk is None:
        row = db.execute(
            "SELECT id FROM chunks WHERE origin=? ORDER BY seq LIMIT 1", (origin,)
        ).fetchone()
        chunk = row[0] if row else None
    if chunk is None:
        return None, False
    start, end = db.execute("SELECT start,end FROM chunks WHERE id=?", (chunk,)).fetchone()
    return source_span(db, origin, start, end, "target_context_unresolved_scope"), False


def pack_groups(groups, config, reference_limit):
    hits, links, counts, used = [], [], {}, 0
    for spans, link in groups:
        additions = []
        for hit in spans:
            origin, start, end = hit["origin_id"], hit["char_start"], hit["char_end"]
            if any(
                h["origin_id"] == origin and h["char_start"] <= start and end <= h["char_end"]
                for h in hits + additions
            ):
                continue
            # Keep exact source spans; partial overlap may be informative for a change bundle.
            additions.append(hit)
        next_links = links + ([link] if link else [])
        ref_chars = len(json.dumps(next_links, ensure_ascii=False)) if next_links else 0
        needed = sum(len(h["quote"]) + len(h["document_header"]) for h in additions)
        old_refs = len(json.dumps(links, ensure_ascii=False)) if links else 0
        if (
            ref_chars > reference_limit
            or used + needed + ref_chars - old_refs > config["context_chars"]
            or len(hits) + len(additions) > config["top_k"]
        ):
            continue
        next_counts = dict(counts)
        for h in additions:
            next_counts[h["origin_id"]] = next_counts.get(h["origin_id"], 0) + 1
        if any(n > config["max_per_document"] for n in next_counts.values()):
            continue
        hits.extend(additions)
        links, counts = next_links, next_counts
        used += needed + ref_chars - old_refs
    return hits, links, used


def retrieve(output, question, *, no_relations=False):
    started = time.monotonic()
    question = base.extract_question(question)
    result = lexical.retrieve(output, question)
    result["amendment_expansion"] = not no_relations
    if no_relations or not result["hits"]:
        return result
    manifest = common.load_build(output, NAME)
    value = manifest["candidate_settings"]
    validate(value)
    cfg = value["amendments"]
    seed_hits = result["hits"][: cfg["seed_hits"]]
    origins = {x["origin_id"] for x in seed_hits}
    db = common.open_index(output)
    try:
        found = {}
        for origin in sorted(origins):
            for (raw,) in db.execute(
                (
                    "SELECT value FROM relations WHERE (source=? OR target=?) AND r"
                    "esolution='unique' ORDER BY id"
                ),
                (origin, origin),
            ):
                relation = json.loads(raw)
                if relation["relation_type"] != "mentions" or cfg["include_mentions"]:
                    found[relation["relation_id"]] = relation
        query_terms = set(base.terms(question))
        ordered = sorted(
            found.values(),
            key=lambda r: (
                -len(query_terms & set(base.terms(r["evidence_quote"]))),
                r["relation_id"],
            ),
        )
        groups: list[tuple[list[Any], dict[str, Any] | None]] = [([hit], None) for hit in seed_hits]
        inspected = 0
        for relation in ordered[: cfg["max_relations"]]:
            inspected += 1
            a, b = relation["evidence_start"], relation["evidence_end"]
            complete = b - a <= cfg["max_span_chars"]
            if not complete:
                a = max(a, relation["reference_start"] - 300)
                b = min(b, a + cfg["max_span_chars"])
            source = source_span(db, relation["source_origin"], a, b, "amendment_basis")
            target, located = target_span(db, relation, question, manifest, cfg["max_span_chars"])
            if target is None:
                continue
            link = {
                k: relation[k]
                for k in (
                    "relation_id",
                    "source_origin",
                    "target_origin",
                    "relation_type",
                    "target_clauses",
                    "target_annexes",
                    "target_role",
                    "resolution",
                    "legal_status",
                )
            }
            link.update(
                evidence_start=a,
                evidence_end=b,
                evidence_complete=complete,
                target_scope_located=located,
                classification_method=relations.VERSION,
            )
            groups.append(([source, target], link))
        groups.extend(([hit], None) for hit in result["hits"][cfg["seed_hits"] :])
        hits, links, used = pack_groups(groups, value["lexical"], cfg["max_reference_chars"])
    finally:
        db.close()
    result.update(
        hits=hits,
        links=links,
        context_chars_used=used,
        relations_considered=len(found),
        relations_inspected=inspected,
        legal_validity_assessed=False,
        seconds=time.monotonic() - started,
        provenance=common.provenance(output, PACKAGE),
    )
    return result


def cli(stage):
    from runtime import main

    main(stage)
