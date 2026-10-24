"""Shared source-span packing, candidate metadata, and compatibility CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import api as base


def settings(candidate: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate": candidate,
        "lexical": base.defaults("bm25_sections"),
        "runtime": {"model_timeout_seconds": 600, "answer_max_tokens": 8192},
    }


def validate_settings(value: dict[str, Any], candidate: str) -> None:
    if value.get("schema_version") != 1 or value.get("candidate") != candidate:
        raise ValueError("settings schema or candidate mismatch")
    base.validate_config(value["lexical"])
    if value["lexical"]["method"] != "bm25_sections" or value["lexical"]["rerank"]:
        raise ValueError("these candidates require lexical BM25 without a reranker")
    for number in value["runtime"].values():
        if type(number) is not int or number <= 0:
            raise ValueError("runtime budgets must be positive integers")


def seal_build(output: Path, manifest: dict[str, Any], value: dict[str, Any], package: Path):
    manifest.update(
        candidate=value["candidate"],
        candidate_settings=value,
        implementation=[p.name for p in sorted(package.glob("*.py"))],
    )
    manifest["files"] = sorted(
        p.name for p in output.iterdir() if p.is_file() and p.name != "build.json"
    )
    base.write_json(output / "build.json", manifest)
    return manifest


def load_build(output: Path, candidate: str):
    manifest = json.loads((output / "build.json").read_text(encoding="utf-8"))
    if manifest.get("candidate") != candidate:
        raise ValueError("index belongs to a different candidate")
    validate_settings(manifest["candidate_settings"], candidate)
    if manifest["config"] != manifest["candidate_settings"]["lexical"]:
        raise ValueError("lexical configuration differs from the frozen candidate settings")
    return manifest


def open_index(output: Path):
    return sqlite3.connect((output / "index.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)


def provenance(output: Path, package: Path):
    return {
        "build_manifest": "build.json",
        "implementation": [p.name for p in sorted(package.glob("*.py"))],
    }


def source_hit(origin, text, chunk_id, score, start, end, section, seed, extra=None):
    if not (0 <= start < end <= len(text)):
        raise ValueError("invalid source span")
    line_start = text.count("\n", 0, start) + 1
    line_end = text.count("\n", 0, end - 1) + 1
    result = {
        "chunk_id": chunk_id,
        "span_id": base.digest(f"{origin}:{start}:{end}".encode()),
        "origin_id": origin,
        "source_relpath": origin + ".md",
        "section": section,
        "char_start": start,
        "char_end": end,
        "line_start": line_start,
        "line_end": line_end,
        "seed_char_start": seed[0],
        "seed_char_end": seed[1],
        "quote": text[start:end],
        "document_header": text[:400],
        "score": float(score),
        "citation": f"[/kb/norm/{origin}.md, {section}; строки {line_start}–{line_end}]",
    }
    if extra:
        result.update(extra)
    return result


def select_hits(db, ranked, manifest, provider=None):
    """Pack literal source spans. Expansion never increases the context budget."""
    config = manifest["config"]
    hits, seen, selected = [], set(), []
    documents, counts = {}, Counter()
    remaining = config["context_chars"]
    for chunk_id, score in ranked:
        row = db.execute(
            "SELECT origin,section,start,end FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone()
        origin, section, start, end = row
        if any(o == origin and a <= start and end <= b for o, a, b in selected):
            continue
        if origin not in documents:
            documents[origin] = db.execute(
                "SELECT body FROM documents WHERE origin=?", (origin,)
            ).fetchone()[0]
        text = documents[origin]
        seed = {"start": start, "end": end, "section": section, "role": "seed"}
        ranges = provider(chunk_id, origin, start, end) if provider else [seed]
        ranges = [
            r
            for r in ranges
            if not any(o == origin and a <= r["start"] and r["end"] <= b for o, a, b in selected)
        ]
        if not ranges:
            continue

        current_header_chars = len(text[:400])

        def fits(items, header_chars=current_header_chars, budget=remaining, used=counts[origin]):
            needed = sum(r["end"] - r["start"] + header_chars for r in items)
            return (
                needed <= budget
                and len(hits) + len(items) <= config["top_k"]
                and used + len(items) <= config["max_per_document"]
            )

        if not fits(ranges):
            ranges = [r for r in ranges if r.get("role") == "table_header"] + [seed]
            if not fits(ranges):
                continue
        # A repeated parent is redundant. Partial overlap falls back to the seed.
        if any(
            o == origin and max(a, r["start"]) < min(b, r["end"])
            for r in ranges
            for o, a, b in selected
        ):
            ranges = [seed]
            if not fits(ranges) or any(
                o == origin and max(a, start) < min(b, end) for o, a, b in selected
            ):
                continue
        for r in ranges:
            body_hash = base.digest(text[r["start"] : r["end"]].encode())
            if body_hash in seen:
                continue
            extra = {k: v for k, v in r.items() if k not in {"start", "end", "section"}}
            hit = source_hit(
                origin,
                text,
                chunk_id,
                score,
                r["start"],
                r["end"],
                r.get("section", section),
                (start, end),
                extra,
            )
            hits.append(hit)
            seen.add(body_hash)
            selected.append((origin, r["start"], r["end"]))
            counts[origin] += 1
            remaining -= len(hit["quote"]) + len(hit["document_header"])
        if len(hits) >= config["top_k"]:
            break
    return hits, config["context_chars"] - remaining


def load_env(path: Path):
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip().removeprefix("export ")
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if (
            not separator
            or not re.fullmatch(r"SC71_[A-Z0-9_]+", key.strip())
            or "$" in value
            or "`" in value
        ):
            raise ValueError(f"unsupported env syntax on line {number}")
        parts = shlex.split(value, comments=True)
        if len(parts) != 1:
            raise ValueError(f"expected one env value on line {number}")
        os.environ[key.strip()] = parts[0]


def cli(stage, package, candidate, build, retrieve):
    parser = argparse.ArgumentParser(description=f"SC-71 {candidate}: standalone local candidate")
    parser.add_argument("--build-dir", required=True, type=Path)
    if stage == "build":
        parser.add_argument("--kb-dir", required=True, type=Path)
        parser.add_argument("--settings", type=Path, default=package / "settings.json")
    else:
        parser.add_argument("--question")
        parser.add_argument("--env-file", type=Path)
        parser.add_argument("--format", choices=("json", "answer"), default="answer")
        parser.add_argument("--answer-mode", choices=("llm", "extractive"), default="llm")
        if candidate == "structure_bm25":
            parser.add_argument("--no-expand", action="store_true")
            parser.add_argument("--no-heading-bonus", action="store_true")
        else:
            parser.add_argument("--no-plan", action="store_true")
            parser.add_argument("--plan-file", type=Path)
    args = parser.parse_args()
    try:
        if stage == "build":
            value = json.loads(args.settings.read_text(encoding="utf-8"))
            result = build(args.kb_dir, args.build_dir, value)
            print(
                json.dumps(
                    {k: result[k] for k in ("candidate", "documents", "chunks", "build_seconds")},
                    ensure_ascii=False,
                )
            )
            return
        if args.env_file:
            load_env(args.env_file)
        question = base.extract_question(
            args.question if args.question is not None else sys.stdin.read()
        )
        if candidate == "structure_bm25":
            result = retrieve(
                args.build_dir,
                question,
                expand=not args.no_expand,
                heading_bonus=not args.no_heading_bonus,
            )
        else:
            if args.no_plan and args.plan_file:
                raise ValueError("no-plan and plan-file are mutually exclusive")
            plan = (
                json.loads(args.plan_file.read_text(encoding="utf-8")) if args.plan_file else None
            )
            result = retrieve(args.build_dir, question, no_plan=args.no_plan, plan=plan)
        validation_error = None
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False))
        else:
            runtime = load_build(args.build_dir, candidate)["candidate_settings"]["runtime"]
            os.environ.setdefault("SC71_CHAT_MAX_TOKENS", str(runtime["answer_max_tokens"]))
            answer = base.answer_question(
                question, result, args.answer_mode, runtime["model_timeout_seconds"], validate=False
            )
            print(answer)
            result["model_calls_generation"] = int(args.answer_mode == "llm")
            if args.answer_mode == "llm":
                try:
                    base.validate_answer(answer, result["hits"])
                except ValueError as error:
                    validation_error = str(error)
                    result["answer_validation_error"] = validation_error
                    result["status"] = "answer_validation_failed"
        trace = {k: v for k, v in result.items() if k not in {"hits", "links", "plan", "queries"}}
        trace["hits"] = [
            {k: h[k] for k in ("chunk_id", "origin_id", "char_start", "char_end")}
            for h in result["hits"]
        ]
        if "plan" in result:
            trace["planner"] = {k: v for k, v in result["plan"].items() if k != "queries"}
        print(json.dumps(trace, ensure_ascii=False), file=sys.stderr)
        if validation_error:
            raise SystemExit(2)
    except (
        ValueError,
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ImportError,
        sqlite3.Error,
    ) as error:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "diagnostics": getattr(error, "diagnostics", None),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None
