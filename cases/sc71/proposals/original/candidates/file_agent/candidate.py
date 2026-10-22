"""A bounded, observation-driven file-reading agent over immutable source text."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidates.file_agent import lexical
from retrieval import api as base
from retrieval import common
from retrieval import documents as catalog

NAME = "file_agent"
PACKAGE = Path(__file__).resolve().parent


class AgentError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def defaults():
    value = lexical.defaults()
    value["agent"] = {
        "max_steps": 4,
        "max_query_chars": 600,
        "max_input_chars": 16000,
        "tool_chars": 6000,
        "total_observation_chars": 24000,
        "max_read_lines": 40,
        "max_read_chars": 4000,
        "tool_results": 6,
        "outline_page": 20,
        "max_tokens": 2048,
        "timeout_seconds": 600,
        "reasoning_effort": "low",
    }
    return value


def validate(value):
    lexical.validate(value)
    cfg = value["agent"]
    for key, number in cfg.items():
        if key != "reasoning_effort" and (type(number) is not int or number <= 0):
            raise ValueError("agent budgets must be positive integers")
    if cfg["max_steps"] > 4 or cfg["max_read_chars"] > cfg["tool_chars"]:
        raise ValueError("agent protocol permits at most four steps and bounded reads")
    if cfg["reasoning_effort"] not in (None, "low", "high", "max"):
        raise ValueError("unknown agent reasoning effort")


def prepare_catalog(kb, output, manifest_path=None):
    return catalog.prepare_catalog(kb, output, manifest_path)


def build(kb, output, value, manifest_path=None):
    validate(value)
    manifest = lexical.build(kb, output, value)
    return finish_preparation(kb, output, manifest, value, manifest_path)


def finish_preparation(kb, output, manifest, value, manifest_path=None):
    """Attach this strategy metadata to an already prepared structural index."""
    try:
        summary, _, _, _ = catalog.add_catalog(kb, output, manifest_path)
        if summary["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("corpus changed during index construction")
        manifest["catalog"] = summary
        manifest["agent_instruction_sha256"] = base.file_digest(PACKAGE / "AGENT.md")
        return common.seal_build(output, manifest, value, PACKAGE)
    except Exception:
        (output / "build.json").unlink(missing_ok=True)
        raise


def pack_evidence(evidence, config):
    hits, used, counts = [], 0, {}
    for item in sorted(enumerate(evidence), key=lambda p: (-p[1]["priority"], p[0])):
        hit = item[1]["hit"]
        origin, start, end = hit["origin_id"], hit["char_start"], hit["char_end"]
        if any(
            h["origin_id"] == origin and max(start, h["char_start"]) < min(end, h["char_end"])
            for h in hits
        ):
            continue
        needed = len(hit["quote"]) + len(hit["document_header"])
        if (
            used + needed > config["context_chars"]
            or counts.get(origin, 0) >= config["max_per_document"]
        ):
            continue
        hits.append(hit)
        used += needed
        counts[origin] = counts.get(origin, 0) + 1
        if len(hits) == config["top_k"]:
            break
    return hits, used


class ToolSession:
    def __init__(self, output, value):
        self.output, self.value = Path(output), value
        self.manifest = common.load_build(self.output, NAME)
        self.db = common.open_index(self.output)
        self.evidence = []
        self.seen = set()
        self.observation_chars = 0

    def close(self):
        self.db.close()

    def document(self, origin):
        if not isinstance(origin, str) or not base.ORIGIN.fullmatch(origin):
            raise ValueError("origin_id must be a catalog identifier, never a path")
        row = self.db.execute("SELECT body FROM documents WHERE origin=?", (origin,)).fetchone()
        if row is None:
            raise ValueError("document not found in this corpus")
        return row[0]

    def execute(self, action, *, initial=False):
        pending = []
        response: dict[str, Any]
        cfg = self.value["agent"]
        try:
            if not isinstance(action, dict) or set(action) != {"tool", "args"}:
                raise ValueError("action requires exactly tool and args")
            tool, args = action["tool"], action["args"]
            if not isinstance(args, dict) or not isinstance(tool, str):
                raise ValueError("tool must be text and args must be an object")
            schemas = {
                "search": ({"query"}, {"query"}),
                "grep": ({"query"}, {"query", "origin_id"}),
                "lookup": (set(), {"number", "date", "title"}),
                "outline": ({"origin_id"}, {"origin_id", "offset"}),
                "read": (
                    {"origin_id", "line_start", "line_end"},
                    {"origin_id", "line_start", "line_end"},
                ),
                "read_span": (
                    {"origin_id", "char_start", "char_end"},
                    {"origin_id", "char_start", "char_end"},
                ),
                "finish": (set(), set()),
            }
            if tool not in schemas:
                raise ValueError("unknown read-only tool")
            required, allowed = schemas[tool]
            if not required <= set(args) <= allowed:
                raise ValueError("invalid tool arguments")
            fingerprint = base.digest(json.dumps(action, sort_keys=True).encode())
            if fingerprint in self.seen:
                response = {"status": "duplicate_action", "tool": tool}
            else:
                self.seen.add(fingerprint)
                if tool == "search":
                    query = args["query"]
                    maximum = cfg["max_input_chars"] if initial else cfg["max_query_chars"]
                    if not isinstance(query, str) or not query.strip() or len(query) > maximum:
                        raise ValueError("search query exceeds its budget or is empty")
                    ranked = base.bm25(
                        self.db, query, self.manifest, self.value["lexical"]["pool_k"]
                    )
                    local = dict(
                        self.manifest,
                        config=dict(self.manifest["config"], top_k=cfg["tool_results"]),
                    )
                    rows, _ = common.select_hits(self.db, ranked, local)
                    response = {"status": "ok", "tool": tool, "hits": rows}
                    pending = [{"hit": row, "priority": 1} for row in rows]
                elif tool == "grep":
                    query = args["query"]
                    if not isinstance(query, str) or not 2 <= len(query) <= 300:
                        raise ValueError("literal search requires 2..300 characters")
                    if "origin_id" in args:
                        self.document(args["origin_id"])
                        sources = self.db.execute(
                            "SELECT origin,body FROM documents WHERE origin=?", (args["origin_id"],)
                        )
                    else:
                        sources = self.db.execute(
                            "SELECT origin,body FROM documents ORDER BY origin"
                        )
                    rows, limited = [], False
                    pattern = re.compile(re.escape(query), re.I)
                    for origin, text in sources:
                        for match in pattern.finditer(text):
                            start, end = (
                                max(0, match.start() - 200),
                                min(len(text), match.end() + 400),
                            )
                            chunk = self.db.execute(
                                "SELECT id FROM chunks WHERE origin=? AND start<? AND end>? "
                                "ORDER BY start LIMIT 1",
                                (origin, end, start),
                            ).fetchone()
                            rows.append(
                                common.source_hit(
                                    origin,
                                    text,
                                    chunk[0] if chunk else None,
                                    0,
                                    start,
                                    end,
                                    "буквальное совпадение",
                                    (match.start(), match.end()),
                                    {"role": "grep"},
                                )
                            )
                            if len(rows) >= cfg["tool_results"]:
                                limited = True
                                break
                        if limited:
                            break
                    response = {
                        "status": "ok",
                        "tool": tool,
                        "hits": rows,
                        "limit_reached": limited,
                    }
                    pending = [{"hit": row, "priority": 1} for row in rows]
                elif tool == "lookup":
                    if not args or any(
                        not isinstance(x, str) or not x.strip() or len(x) > 300
                        for x in args.values()
                    ):
                        raise ValueError("lookup requires bounded number, date, or title strings")
                    conditions, params = [], []
                    for key in ("number", "date"):
                        if key in args:
                            conditions.append(key + "=?")
                            params.append(
                                catalog.number_key(args[key]) if key == "number" else args[key]
                            )
                    query = "SELECT value FROM catalog"
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                    query += " ORDER BY origin"
                    rows = []
                    for (raw,) in self.db.execute(query, params):
                        record = json.loads(raw)
                        if (
                            "title" in args
                            and args["title"].casefold() not in record["title"].casefold()
                        ):
                            continue
                        rows.append(record)
                        if len(rows) > cfg["tool_results"]:
                            break
                    response = {
                        "status": "ok",
                        "tool": tool,
                        "records": rows[: cfg["tool_results"]],
                        "more": len(rows) > cfg["tool_results"],
                    }
                elif tool == "outline":
                    origin = args["origin_id"]
                    self.document(origin)
                    offset = args.get("offset", 0)
                    if type(offset) is not int or offset < 0:
                        raise ValueError("outline offset must be nonnegative")
                    rows = self.db.execute(
                        (
                            "SELECT id,parent,kind,label,start,end FROM nodes WHERE"
                            " origin=? ORDER BY start,id LIMIT ? OFFSET ?"
                        ),
                        (origin, cfg["outline_page"] + 1, offset),
                    ).fetchall()
                    response = {
                        "status": "ok",
                        "tool": tool,
                        "origin_id": origin,
                        "nodes": [
                            dict(
                                zip(
                                    ("id", "parent", "kind", "label", "char_start", "char_end"),
                                    row,
                                    strict=True,
                                )
                            )
                            for row in rows[: cfg["outline_page"]]
                        ],
                        "next_offset": offset + cfg["outline_page"]
                        if len(rows) > cfg["outline_page"]
                        else None,
                    }
                elif tool in ("read", "read_span"):
                    origin = args["origin_id"]
                    text = self.document(origin)
                    if tool == "read":
                        a, b = args["line_start"], args["line_end"]
                        lines = catalog.source_lines(text)
                        if (
                            type(a) is not int
                            or type(b) is not int
                            or not 1 <= a <= b <= len(lines)
                            or b - a + 1 > cfg["max_read_lines"]
                        ):
                            raise ValueError(
                                "line range is out of bounds or exceeds the read budget"
                            )
                        start, end = sum(map(len, lines[: a - 1])), sum(map(len, lines[:b]))
                    else:
                        start, end = args["char_start"], args["char_end"]
                        if (
                            type(start) is not int
                            or type(end) is not int
                            or not 0 <= start < end <= len(text)
                        ):
                            raise ValueError("character range is out of bounds")
                    if end - start > cfg["max_read_chars"]:
                        raise ValueError(
                            "requested span exceeds the read budget; request a smaller span"
                        )
                    row = self.db.execute(
                        "SELECT id FROM chunks WHERE origin=? AND start<? AND end>? "
                        "ORDER BY start LIMIT 1",
                        (origin, end, start),
                    ).fetchone()
                    hit = common.source_hit(
                        origin,
                        text,
                        row[0] if row else None,
                        0,
                        start,
                        end,
                        "прочитанный участок",
                        (start, end),
                        {"role": "agent_read"},
                    )
                    response = {"status": "ok", "tool": tool, "hits": [hit]}
                    pending = [{"hit": hit, "priority": 2}]
                else:
                    response = {"status": "finished", "tool": tool}
        except (ValueError, KeyError, TypeError) as error:
            response = {"status": "tool_error", "error": str(error)}
        limit = min(cfg["tool_chars"], cfg["total_observation_chars"] - self.observation_chars)
        # Remove whole entries, never silently truncate a source quote.
        for field in ("hits", "records", "nodes"):
            while len(json.dumps(response, ensure_ascii=False)) > limit and response.get(field):
                response[field].pop()
                response["truncated"] = True
        if len(json.dumps(response, ensure_ascii=False)) > limit:
            response = {"status": "observation_budget"}
            pending = []
        if "hits" in response:
            shown = {hit["span_id"] for hit in response["hits"]}
            self.evidence.extend(x for x in pending if x["hit"]["span_id"] in shown)
        size = len(json.dumps(response, ensure_ascii=False))
        # The caller stops before another tool/model action when fewer than 256 remain.
        if size > limit:
            raise ValueError("observation budget exhausted")
        self.observation_chars += size
        return response


def retrieve(output, question, *, no_agent=False):
    started = time.monotonic()
    question = base.extract_question(question)
    manifest = common.load_build(output, NAME)
    value = manifest["candidate_settings"]
    validate(value)
    if no_agent:
        result = lexical.retrieve(output, question)
        result["agent"] = {"enabled": False, "stop_reason": "disabled", "steps": []}
        return result
    if len(question) > value["agent"]["max_input_chars"]:
        raise ValueError("question exceeds agent input budget")
    instruction = (PACKAGE / "AGENT.md").read_text(encoding="utf-8")
    if base.digest(instruction.encode()) != manifest["agent_instruction_sha256"]:
        raise ValueError("agent instruction changed after this build")
    url, model = os.environ.get("SC71_CHAT_URL", ""), os.environ.get("SC71_CHAT_MODEL", "")
    if not url or not model:
        raise ValueError("file agent requires SC71_CHAT_URL and SC71_CHAT_MODEL")
    session = ToolSession(output, value)
    calls, steps, stop = [], [], "step_limit"
    try:
        initial = session.execute({"tool": "search", "args": {"query": question}}, initial=True)
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "initial_search": initial}, ensure_ascii=False
                ),
            },
        ]
        for index in range(value["agent"]["max_steps"]):
            if value["agent"]["total_observation_chars"] - session.observation_chars < 256:
                stop = "observation_budget"
                break
            cfg = value["agent"]
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": cfg["max_tokens"],
                "messages": messages,
            }
            if cfg["reasoning_effort"] is not None:
                payload["reasoning_effort"] = cfg["reasoning_effort"]
            call = {"step": index + 1, "model": model, "usage": None}
            calls.append(call)
            raw = None
            tick = time.monotonic()
            try:
                data = base.post_json(url, payload, "SC71_CHAT_KEY", cfg["timeout_seconds"])
                call["usage"] = data.get("usage")
                raw = data["choices"][0]["message"]["content"]
                if not isinstance(raw, str) or not raw.strip() or len(raw) > 16384:
                    raise ValueError("empty or oversized agent reply")
                action = json.loads(raw)
                if not isinstance(action, dict) or set(action) != {"tool", "args"}:
                    raise ValueError("agent reply must be an action object")
                call["response_sha256"] = base.digest(raw.encode())
                call["finish_reason"] = data["choices"][0].get("finish_reason")
            except (ValueError, KeyError, IndexError, TypeError, OSError):
                raise AgentError(
                    "agent model call failed; no replacement action was generated",
                    {
                        "phase": "agent",
                        "model_calls_retrieval": len(calls),
                        "calls": calls,
                        "steps": steps,
                        "observation_chars": session.observation_chars,
                    },
                ) from None
            finally:
                call["seconds"] = time.monotonic() - tick
            observation = session.execute(action)
            steps.append(
                {
                    "step": index + 1,
                    "action": action,
                    "status": observation["status"],
                    "observation_sha256": base.digest(
                        json.dumps(observation, sort_keys=True).encode()
                    ),
                }
            )
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": json.dumps(observation, ensure_ascii=False)},
                ]
            )
            if observation["status"] in ("finished", "observation_budget"):
                stop = "finish" if observation["status"] == "finished" else "observation_budget"
                break
        hits, used = pack_evidence(session.evidence, value["lexical"])
        observation_chars = session.observation_chars
    finally:
        session.close()
    return {
        "method": NAME,
        "status": "retrieved" if hits else "no_candidates",
        "hits": hits,
        "links": [],
        "corpus_sha256": manifest["corpus_sha256"],
        "query_sha256": base.digest(question.encode()),
        "agent": {
            "enabled": True,
            "stop_reason": stop,
            "steps": steps,
            "calls": calls,
            "observation_chars": observation_chars,
            "instruction_sha256": manifest["agent_instruction_sha256"],
        },
        "model_calls_retrieval": len(calls),
        "context_chars_used": used,
        "seconds": time.monotonic() - started,
        "provenance": common.provenance(output, PACKAGE),
    }


def cli(stage):
    from runtime import main

    main(stage)
