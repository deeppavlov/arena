"""Conservative, typed references. Resolved text is not a legal-validity verdict."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from retrieval import api as base
from retrieval import documents as catalog
from retrieval.chunking import parse_document

VERSION = "explicit-amendments-v1"
OPERATIVE = re.compile(
    r"\b(внести|изложить|дополнить|исключить|признать|отменить|отменять)\b", re.I
)
CLAUSE_REF = re.compile(r"\b(?:пункт[а-яё]*|подпункт[а-яё]*)\s+[«\"“]?(\d+(?:\.\d+){0,5})", re.I)
ANNEX_REF = re.compile(r"\bприложени[а-яё]*\s*№?\s*(\d+)", re.I)
ADMIN_TAIL = re.compile(r"(?m)^\s*(?:ЛИСТ\s+СОГЛАСОВАНИЯ|Ректор\b|Инициатор\s+согласования)", re.I)


def relation_kind(unit):
    for match in OPERATIVE.finditer(unit):
        if re.search(r"\bне\s+$", unit[max(0, match.start() - 20) : match.start()], re.I):
            continue
        word = match[0].casefold()
        after = unit[match.end() :]
        if word == "признать" and re.search(r"утративш[а-яё]*\s+силу", after, re.I):
            return "repeals"
        if word.startswith("отмен"):
            return "repeals"
        if word in ("изложить", "дополнить", "исключить"):
            return "amends"
        if word == "внести" and re.search(r"изменени", after, re.I):
            return "amends"
    return "mentions"


def identity_key(identity):
    return identity["kind"], identity["date"], catalog.number_key(identity["number"])


def extract_for_document(origin, text, parsed, identities):
    seen = set()
    matches = sorted(
        [
            m
            for pattern in (catalog.REFERENCE, catalog.REFERENCE_REVERSE)
            for m in pattern.finditer(text)
        ],
        key=lambda m: m.start(),
    )
    for match in matches:
        if match.span() in seen or not catalog.valid_date(match["date"]):
            continue
        seen.add(match.span())
        reference = {
            "kind": catalog.kind_key(match["kind"]),
            "date": match["date"],
            "number": match["number"],
        }
        candidates = sorted(identities.get(identity_key(reference), []))
        # The act's own designation is not an external legal link.
        if candidates == [origin]:
            continue
        candidates = [x for x in candidates if x != origin]
        containers = [
            n
            for n in parsed["nodes"]
            if n["kind"] == "clause" and n["start"] <= match.start() and match.end() <= n["end"]
        ]
        if containers:
            node = min(containers, key=lambda n: n["end"] - n["start"])
            start, end = node["start"], node["end"]
        else:
            start = text.rfind("\n", 0, match.start()) + 1
            newline = text.find("\n", match.end())
            end = newline + 1 if newline >= 0 else len(text)
        whole = text[start:end]
        administrative = next(ADMIN_TAIL.finditer(whole), None)
        outside_operative = (
            administrative is not None and match.start() - start >= administrative.start()
        )
        if administrative is not None and not outside_operative:
            end = start + administrative.start()
            whole = text[start:end]
        # Do not apply an imperative from a previous sentence to a mere later mention.
        boundary = re.compile(
            (
                "[.;]\\s+(?=(?:Учесть|Учитывать|Руководствоваться|Внести|Признать|От"
                "менить|Изложить)\\b)"
            ),
            re.I,
        )
        cuts = [0, *(m.end() for m in boundary.finditer(whole)), len(whole)]
        relative = match.start() - start
        segment = next(
            whole[a:b] for a, b in zip(cuts, cuts[1:], strict=False) if a <= relative < b
        )
        segment_offset = text.find(segment, start, end)
        local_position = match.start() - segment_offset
        window = segment[max(0, local_position - 600) : local_position + len(match[0]) + 400]
        kind = "mentions" if outside_operative else relation_kind(window)
        # Multiple references in one operative segment need a richer scope parser.
        references_in_segment = sum(
            len(list(p.finditer(segment))) for p in (catalog.REFERENCE, catalog.REFERENCE_REVERSE)
        )
        action_scope_ambiguous = kind != "mentions" and references_in_segment > 1
        if action_scope_ambiguous:
            kind = "mentions"
        clauses = list(dict.fromkeys(CLAUSE_REF.findall(segment))) if kind != "mentions" else []
        annexes = list(dict.fromkeys(ANNEX_REF.findall(segment))) if kind != "mentions" else []
        approval = bool(
            re.search(
                r"утвержд[а-яё]*\s*$", text[max(start, match.start() - 100) : match.start()], re.I
            )
        )
        # A large clause is retained as coordinates; retrieval will expose a bounded window.
        quote = text[start:end]
        resolution = "unique" if len(candidates) == 1 else "ambiguous" if candidates else "missing"
        yield {
            "schema_version": 1,
            "relation_id": base.digest(f"{origin}:{match.start()}:{match.end()}".encode()),
            "source_origin": origin,
            "target_origin": candidates[0] if resolution == "unique" else None,
            "target_candidates": candidates,
            "resolution": resolution,
            "reference": reference,
            "reference_start": match.start(),
            "reference_end": match.end(),
            "relation_type": kind,
            "target_clauses": clauses,
            "target_annexes": annexes,
            "target_role": "approval_act" if approval else "referenced_act",
            "scope": "clauses" if clauses else "unspecified",
            "evidence_start": start,
            "evidence_end": end,
            "evidence_quote": quote,
            "legal_status": "unknown",
            "classification_method": VERSION,
            "action_scope_ambiguous": action_scope_ambiguous,
            "limitations": (
                "textual rule and matching requisites; issuer, effect, dates, and c"
                "ompleteness unverified"
            ),
        }


def build_relations(rows, kb, output, db=None):
    kb, _ = catalog.corpus_paths(kb)
    identities = defaultdict(list)
    for row in rows:
        if row["identity"]:
            identities[identity_key(row["identity"])].append(row["origin_id"])
    relations, types, resolutions = [], Counter(), Counter()
    with (output / "relations.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            origin = row["origin_id"]
            text = (kb / (origin + ".md")).read_bytes().decode("utf-8-sig")
            if base.file_digest(kb / (origin + ".md")) != row["sha256"]:
                raise ValueError("corpus changed during relation extraction")
            parsed = parse_document(text)
            for relation in extract_for_document(origin, text, parsed, identities):
                relations.append(relation)
                stream.write(json.dumps(relation, ensure_ascii=False) + "\n")
                types[relation["relation_type"]] += 1
                resolutions[relation["resolution"]] += 1
    summary = {
        "version": VERSION,
        "relations": len(relations),
        "types": dict(types),
        "resolutions": dict(resolutions),
        "model_calls": 0,
        "legal_validity_assessed": False,
        "manual_review_completed": False,
    }
    base.write_json(output / "relations-summary.json", summary)
    if db is not None:
        db.execute(
            "CREATE TABLE relations(id TEXT PRIMARY KEY, source TEXT, target TE"
            "XT, kind TEXT, resolution TEXT, value TEXT)"
        )
        db.execute("CREATE INDEX relation_source ON relations(source)")
        db.execute("CREATE INDEX relation_target ON relations(target)")
        db.executemany(
            "INSERT INTO relations VALUES (?,?,?,?,?,?)",
            [
                (
                    x["relation_id"],
                    x["source_origin"],
                    x["target_origin"],
                    x["relation_type"],
                    x["resolution"],
                    json.dumps(x, ensure_ascii=False),
                )
                for x in relations
            ],
        )
        db.commit()
    return summary
