"""Local build/run CLI, common answer generator, and separate diagnostic JSON."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from . import api as base
from . import common


def diagnostic(result):
    trace = {k: v for k, v in result.items() if k not in ("hits", "links", "agent")}
    trace["hits"] = [
        {k: h[k] for k in ("chunk_id", "origin_id", "char_start", "char_end")}
        for h in result["hits"]
    ]
    trace["links"] = result["links"]
    if "agent" in result:
        agent = result["agent"]
        trace["agent"] = {k: v for k, v in agent.items() if k != "steps"}
        trace["agent"]["steps"] = [
            {k: v for k, v in step.items() if k != "action"}
            | {
                "tool": step["action"].get("tool"),
            }
            for step in agent["steps"]
        ]
    return trace


def main(stage, candidate):
    parser = argparse.ArgumentParser(
        description=f"SC-71 {candidate.NAME}: standalone local candidate"
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    if stage == "build":
        parser.add_argument("--kb-dir", type=Path, required=True)
        parser.add_argument("--source-manifest", type=Path)
        parser.add_argument("--settings", type=Path, default=candidate.PACKAGE / "settings.json")
        parser.add_argument("--catalog-only", action="store_true")
    else:
        parser.add_argument("--question")
        parser.add_argument("--env-file", type=Path)
        parser.add_argument("--format", choices=("json", "answer"), default="answer")
        parser.add_argument("--answer-mode", choices=("llm", "extractive"), default="llm")
        parser.add_argument(
            "--no-agent" if candidate.NAME == "file_agent" else "--no-relations",
            action="store_true",
        )
    args = parser.parse_args()
    result = None
    try:
        if stage == "build":
            value = json.loads(args.settings.read_text(encoding="utf-8"))
            candidate.validate(value)
            if args.catalog_only:
                summary = candidate.prepare_catalog(
                    args.kb_dir, args.build_dir, args.source_manifest
                )
                print(json.dumps(summary, ensure_ascii=False))
            else:
                manifest = candidate.build(args.kb_dir, args.build_dir, value, args.source_manifest)
                print(
                    json.dumps(
                        {
                            k: manifest[k]
                            for k in (
                                "candidate",
                                "documents",
                                "chunks",
                                "build_seconds",
                                "catalog",
                            )
                        },
                        ensure_ascii=False,
                    )
                )
            return
        if args.env_file:
            common.load_env(args.env_file)
        question = base.extract_question(
            args.question if args.question is not None else sys.stdin.read()
        )
        options = (
            {"no_agent": args.no_agent}
            if candidate.NAME == "file_agent"
            else {"no_relations": args.no_relations}
        )
        result = candidate.retrieve(args.build_dir, question, **options)
        failed = False
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False))
        else:
            runtime = common.load_build(args.build_dir, candidate.NAME)["candidate_settings"][
                "runtime"
            ]
            os.environ.setdefault("SC71_CHAT_MAX_TOKENS", str(runtime["answer_max_tokens"]))
            started = time.monotonic()
            result["model_calls_generation"] = 0
            post = base.post_json

            def capture(*params, **kwargs):
                result["model_calls_generation"] += 1
                data = post(*params, **kwargs)
                result["usage_generation"] = data.get("usage")
                return data

            base.post_json = capture
            try:
                answer = base.answer_question(
                    question,
                    result,
                    args.answer_mode,
                    runtime["model_timeout_seconds"],
                    validate=False,
                )
            finally:
                base.post_json = post
                result["generation_seconds"] = time.monotonic() - started
            print(answer)
            if args.answer_mode == "llm":
                try:
                    base.validate_answer(answer, result["hits"])
                except ValueError as error:
                    result["status"] = "answer_validation_failed"
                    result["answer_validation_error"] = str(error)
                    failed = True
        print(json.dumps(diagnostic(result), ensure_ascii=False), file=sys.stderr)
        if failed:
            raise SystemExit(2)
    except (ValueError, OSError, KeyError, IndexError, TypeError, sqlite3.Error) as error:
        detail = getattr(error, "diagnostics", None)
        if detail and "steps" in detail:
            # Keep error traces without source/query text, just like successful stderr traces.
            detail = dict(detail)
            detail["steps"] = [
                {k: v for k, v in step.items() if k != "action"} for step in detail["steps"]
            ]
        print(
            json.dumps(
                {
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "diagnostics": detail,
                    "partial_result": diagnostic(result) if result else None,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None
