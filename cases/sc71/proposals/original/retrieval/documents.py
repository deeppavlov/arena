"""Read-only corpus inspection and a portable, evidence-backed document catalog."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import storage as base
from .chunking import metadata, parse_document
from .config import DESIGNATION, SOURCE_LINK

VERSION = "catalog-v1"
DATE = r"\d{2}\.\d{2}\.\d{4}"
NUMBER = r"[\w]+(?:[/\-–—][\w]+)*"
KIND = r"приказ[а-яё]*|распоряжени[а-яё]*|положени[а-яё]*|регламент[а-яё]*|поряд[а-яё]*"
REFERENCE = re.compile(
    rf"\b(?P<kind>{KIND})\s+(?:[А-ЯЁA-Z][\w.-]*\s+)?от\s+(?P<date>{DATE})"
    rf"\s*(?:г(?:ода)?\.?)?\s*№\s*(?P<number>{NUMBER})",
    re.I,
)
REFERENCE_REVERSE = re.compile(
    rf"\b(?P<kind>{KIND})\s+№\s*(?P<number>{NUMBER})\s+от\s+(?P<date>{DATE})", re.I
)
ACT = re.compile(r"^\s*#*\s*(?P<kind>приказ|распоряжение|положение|регламент|порядок)\b", re.I)


def source_lines(text):
    """LF defines source line coordinates; CR and page separators stay in the text."""
    pieces = text.split("\n")
    return [piece + "\n" for piece in pieces[:-1]] + ([pieces[-1]] if pieces[-1] else [])


def kind_key(value):
    for prefix, key in (
        ("приказ", "order"),
        ("распоряжени", "directive"),
        ("положени", "regulation"),
        ("регламент", "regulation"),
        ("поряд", "procedure"),
    ):
        if value.casefold().startswith(prefix):
            return key
    return "unknown"


def number_key(value):
    return value.casefold().replace("–", "-").replace("—", "-")


def valid_date(value):
    try:
        datetime.strptime(value, "%d.%m.%Y")
    except ValueError:
        return False
    return True


def primary_identity(text):
    """Only an act heading and its own short header; body references are excluded."""
    lines = text.splitlines(keepends=True)
    offset, chosen = 0, None
    for line in lines:
        if offset > 1200:
            break
        match = ACT.match(line)
        if match:
            chosen = (offset, match, line)
            break
        offset += len(line)
    if chosen is None:
        return None, {}
    start, match, _ = chosen
    cursor, stop = start, start
    for line in text[start : start + 800].splitlines(keepends=True):
        # A subject line, a clause, an approval signature, or another act ends the header.
        clean = line.strip().lstrip("# ")
        if cursor > start and (
            re.match(r"(?:Об?\s|\d+(?:\.\d+)*[.)]\s|УТВЕРЖД|Приложени)", clean, re.I)
            or ACT.match(line)
        ):
            break
        stop = cursor + len(line)
        cursor = stop
    header = text[start:stop]
    number = re.search(rf"№\s*({NUMBER})", header)
    date = re.search(DATE, header)
    if not number or not date or not valid_date(date[0]):
        return None, {}
    identity = {"kind": kind_key(match["kind"]), "number": number[1], "date": date[0]}
    spans = {
        "number": [start + number.start(1), start + number.end(1)],
        "date": [start + date.start(), start + date.end()],
    }
    return identity, spans


def corpus_paths(kb):
    kb = Path(kb).resolve()
    if (kb / "norm").is_dir():
        kb /= "norm"
    paths = sorted(kb.glob("*.md"))
    if not paths or any(not base.ORIGIN.fullmatch(p.stem) or p.is_symlink() for p in paths):
        raise ValueError("corpus must contain regular 64-hex named Markdown files")
    return kb, paths


def snapshot(kb):
    """Read source bytes only; paths in metadata are never opened."""
    kb, paths = corpus_paths(kb)
    sources = [p.stem for p in paths]
    cache_key = hashlib.sha256()
    for path in paths:
        cache_key.update(f"{path.stem}:{path.stat().st_size}:".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                cache_key.update(block)
    return kb, sources, cache_key.hexdigest()


def source_metadata(kb, manifest_path=None):
    source = Path(manifest_path) if manifest_path else kb.parent / "manifest.json"
    if not source.exists() and manifest_path is None:
        return {}, {"source_manifest": None, "corpus_version": None, "snapshot_tag": None}
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    mapping = {}
    for item in data.get("objects", []):
        origin, relative = item.get("origin_id"), item.get("relative_posix")
        if not isinstance(origin, str) or not base.ORIGIN.fullmatch(origin):
            raise ValueError("invalid origin identifier in the source manifest")
        if origin in mapping and mapping[origin] != relative:
            raise ValueError("conflicting original paths in the source manifest")
        if relative is not None and not isinstance(relative, str):
            raise ValueError("original path must be text")
        mapping[origin] = relative  # Display metadata only; never open this path.
    return mapping, {
        "source_manifest": source.name,
        "corpus_version": data.get("corpus_version"),
        "snapshot_tag": data.get("snapshot_tag"),
    }


def records(kb, manifest_path=None):
    kb, paths = corpus_paths(kb)
    mapping, source = source_metadata(kb, manifest_path)
    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        meta = metadata(text)
        identity, spans = primary_identity(text)
        # Do not advertise the first referenced order number as the document's own number.
        if identity:
            meta.update(number=identity["number"], date=identity["date"])
        else:
            meta.update(number=None, date=None)
        parsed = parse_document(text)
        record = {
            "schema_version": 1,
            "origin_id": path.stem,
            "source_relpath": path.name,
            "original_path": mapping.get(path.stem),
            "characters": len(text),
            "lines": len(source_lines(text)),
            "title": meta["title"],
            "doc_type": meta["doc_type"],
            "identity": identity,
            "identity_spans": spans,
            "title_span": meta["metadata_spans"]["title"],
            "node_count": len(parsed["nodes"]),
            "legal_status": "unknown",
            "metadata_method": VERSION + "; header heuristics; not a legal decision",
        }
        yield record, text, parsed, source


def write_catalog(kb, output, manifest_path=None, db=None):
    output = Path(output)
    rows, kinds = [], Counter()
    source = {}
    for record, _text, _parsed, current_source in records(kb, manifest_path):
        source = current_source
        rows.append(record)
        kinds[record["doc_type"]] += 1
    with (output / "catalog.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in rows:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "catalog_version": VERSION,
        "documents": len(rows),
        "identified": sum(x["identity"] is not None for x in rows),
        "with_original_path": sum(x["original_path"] is not None for x in rows),
        "profiles": dict(kinds),
        "model_calls": 0,
        **source,
    }
    base.write_json(output / "catalog-summary.json", summary)
    if db is not None:
        db.execute(
            "CREATE TABLE catalog(origin TEXT PRIMARY KEY, number TEXT, date TEXT, value TEXT)"
        )
        db.execute("CREATE INDEX catalog_number ON catalog(number,date)")
        db.executemany(
            "INSERT INTO catalog VALUES (?,?,?,?)",
            [
                (
                    x["origin_id"],
                    number_key(x["identity"]["number"]) if x["identity"] else None,
                    x["identity"]["date"] if x["identity"] else None,
                    json.dumps(x, ensure_ascii=False),
                )
                for x in rows
            ],
        )
        db.commit()
    return summary, rows, {}, {}


def prepare_catalog(kb, output, manifest_path=None):
    kb, _ = corpus_paths(kb)
    output = Path(output).resolve()
    if kb == output or kb in output.parents or output in kb.parents:
        raise ValueError("catalog must be outside the source corpus")
    if output.exists() and any(output.iterdir()):
        raise ValueError("catalog directory must be empty; use a new version")
    output.mkdir(parents=True, exist_ok=True)
    return write_catalog(kb, output, manifest_path)[0]


def add_catalog(kb, output, manifest_path=None):
    db = sqlite3.connect(output / "index.sqlite3")
    try:
        return write_catalog(kb, output, manifest_path, db)
    finally:
        db.close()


def extract_links(docs: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    headers = defaultdict(set)
    known = {origin for origin, _ in docs}
    for origin, text in docs:
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        for match in DESIGNATION.finditer(first_line):
            key = (
                match["kind"].lower(),
                match["date"] or match["date2"],
                (match["num"] or match["num2"]).lower(),
            )
            headers[key].add(origin)
    edges = []
    stats = {"ambiguous": 0, "unresolved": 0}
    for origin, text in docs:
        for match in DESIGNATION.finditer(text):
            key = (
                match["kind"].lower(),
                match["date"] or match["date2"],
                (match["num"] or match["num2"]).lower(),
            )
            candidates = headers.get(key, set())
            if len(candidates) > 1:
                stats["ambiguous"] += 1
                continue
            if not candidates:
                stats["unresolved"] += 1
                continue
            target = next(iter(candidates))
            if target != origin:
                edges.append(
                    {
                        "source": origin,
                        "target": target,
                        "quote": match[0],
                        "offset": match.start(),
                        "type": "explicit_designation",
                    }
                )
        for match in SOURCE_LINK.finditer(text):
            target = match[1]
            if target in known and target != origin:
                edges.append(
                    {
                        "source": origin,
                        "target": target,
                        "quote": match[0],
                        "offset": match.start(),
                        "type": "explicit_path",
                    }
                )
    return edges, stats
