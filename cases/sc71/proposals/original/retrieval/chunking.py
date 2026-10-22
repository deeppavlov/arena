"""Heuristic document trees. Coordinates always address the unmodified source."""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

from .config import BOUNDARY

HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})([.)]?)\s+(\S.*)$")
ROMAN = re.compile(r"^\s*([IVXLCDM]+)[.)]\s+(.{3,120})$", re.I)
TABLE_SEPARATOR = re.compile(r"^[\s|:\-]+$")


def metadata(text: str) -> dict[str, Any]:
    header = text[:5000]
    title = ""
    kind = "generic"
    patterns = (
        ("order", r"(?:^|\n)\s*#*\s*(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)\b"),
        ("contract", r"(?:^|\n)\s*#*\s*(?:ДОГОВОР|СОГЛАШЕНИЕ)\b"),
        ("regulation", r"(?:^|\n)\s*#*\s*(?:ПОЛОЖЕНИЕ|РЕГЛАМЕНТ|ПОРЯДОК|ИНСТРУКЦИЯ)\b"),
    )
    kind_span = None
    for candidate, pattern in patterns:
        match = re.search(pattern, header, re.I)
        if match:
            kind = candidate
            kind_span = list(match.span())
            break
    for line in header.splitlines():
        clean = line.strip().lstrip("# ")
        if re.match(r"(?:Об?\s+|Положение\b|Регламент\b|Договор\b|Порядок\b)", clean, re.I):
            title = clean[:300]
            break
    if not title:
        title = next(
            (line.strip().lstrip("# ")[:300] for line in header.splitlines() if line.strip()), ""
        )
    number = re.search(r"№\s*([\w/-]+)", header)
    date = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", header)
    return {
        "doc_type": kind,
        "title": title,
        "number": number[1] if number else None,
        "date": date[0] if date else None,
        "metadata_spans": {
            "doc_type_hint": kind_span,
            "title": [header.find(title), header.find(title) + len(title)] if title else None,
            "number": list(number.span(1)) if number else None,
            "date": list(date.span()) if date else None,
        },
        "metadata_method": "header heuristics; no assertion of legal validity",
    }


def is_table_row(line: str) -> bool:
    return line.count("|") >= 2 or line.count("\t") >= 2


def clause_label(line: str):
    match = CLAUSE.match(line)
    if not match:
        return None
    label, punctuation, rest = match.groups()
    parts = label.split(".")
    # Dates, standalone page numbers, and bare year-prefixed prose are not clauses.
    if int(parts[0]) > 999 or (len(parts) == 3 and len(parts[-1]) == 4):
        return None
    if (
        not punctuation
        and "." not in label
        and not re.match(
            r"(?:Общие|Основные|Заключительные|Порядок|Права|Обязанности)\b", rest, re.I
        )
    ):
        return None
    return label


def windows(text: str, start: int, end: int, limit: int, table: bool = False):
    while start < end:
        stop = min(end, start + limit)
        if stop < end:
            paragraphs = list(re.finditer(r"\n[ \t\r]*\n", text[start:stop])) if not table else []
            newline = text.rfind("\n", start + max(1, limit // 3), stop)
            if paragraphs and paragraphs[-1].end() >= limit // 3:
                stop = start + paragraphs[-1].end()
            elif newline >= 0:
                stop = newline + 1
            elif not table:
                space = text.rfind(" ", start + max(1, limit // 3), stop)
                if space >= 0:
                    stop = space + 1
        if text[start:stop].strip():
            yield start, stop
        start = stop


def parse_document(text: str, limit: int = 1800) -> dict[str, Any]:
    if type(limit) is not int or limit < 128:
        raise ValueError("chunk limit must be at least 128 characters")
    if "\x00" in text:
        raise ValueError("NUL byte in source document")
    meta = metadata(text)
    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    nodes = [
        {
            "id": 0,
            "parent": None,
            "kind": "document",
            "label": meta["title"],
            "start": 0,
            "end": len(text),
            "children": [],
            "level": 0,
            "header_end": None,
        }
    ]
    headings: list[int] = []
    clauses: list[int] = []
    lines = []
    offset = 0

    def close(stack, position):
        nodes[stack.pop()]["end"] = position

    def append(parent, kind, label, start, level=0):
        identifier = len(nodes)
        nodes.append(
            {
                "id": identifier,
                "parent": parent,
                "kind": kind,
                "label": label,
                "start": start,
                "end": len(text),
                "children": [],
                "level": level,
                "header_end": None,
            }
        )
        nodes[parent]["children"].append(identifier)
        return identifier

    for line in text.splitlines(keepends=True):
        lines.append((offset, offset + len(line), line))
        heading = HEADING.match(line)
        roman = ROMAN.match(line) if meta["doc_type"] != "generic" else None
        if heading or roman:
            while clauses:
                close(clauses, offset)
            if heading:
                level, label = len(heading[1]), heading[2]
            else:
                assert roman is not None
                level, label = 2, roman[0].strip()
            while headings and nodes[headings[-1]]["level"] >= level:
                close(headings, offset)
            parent = headings[-1] if headings else 0
            headings.append(append(parent, "section", label, offset, level))
        elif meta["doc_type"] != "generic" and not is_table_row(line):
            label = clause_label(line)
            if label:
                while clauses and not label.startswith(nodes[clauses[-1]]["label"] + "."):
                    close(clauses, offset)
                parent = clauses[-1] if clauses else (headings[-1] if headings else 0)
                clauses.append(append(parent, "clause", label, offset))
        offset += len(line)
    while clauses:
        close(clauses, len(text))
    while headings:
        close(headings, len(text))

    # Tables remain children of their actual containing section/clause.
    position = 0
    while position < len(lines):
        if not is_table_row(lines[position][2]):
            position += 1
            continue
        final = position + 1
        while final < len(lines) and is_table_row(lines[final][2]):
            final += 1
        start, end = lines[position][0], lines[final - 1][1]
        containers = [n for n in nodes if n["start"] <= start and end <= n["end"]]
        parent = min(containers, key=lambda n: (n["end"] - n["start"], -n["id"]))["id"]
        table_id = append(parent, "table", "таблица", start)
        nodes[table_id]["end"] = end
        header_end = lines[position][1]
        if position + 1 < final and TABLE_SEPARATOR.fullmatch(lines[position + 1][2].strip()):
            header_end = lines[position + 1][1]
        nodes[table_id]["header_end"] = header_end
        position = final

    chunks = []
    for node in nodes:
        children = sorted((nodes[i] for i in node["children"]), key=lambda n: n["start"])
        own = []
        cursor = node["start"]
        for child in children:
            if cursor < child["start"]:
                own.append((cursor, child["start"]))
            cursor = max(cursor, child["end"])
        if cursor < node["end"]:
            own.append((cursor, node["end"]))
        ancestors = []
        parent = node["id"]
        while parent is not None:
            ancestors.append(nodes[parent]["label"])
            parent = nodes[parent]["parent"]
        path = list(reversed([a for a in ancestors if a]))
        for left, right in own:
            for start, end in windows(text, left, right, limit, node["kind"] == "table"):
                chunks.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "line_start": bisect_right(line_starts, start),
                        "line_end": bisect_right(line_starts, end - 1),
                        "section": " / ".join(path[-3:]) or "раздел документа",
                        "node_id": node["id"],
                        "heading_path": path,
                    }
                )
    chunks.sort(key=lambda c: (c["start"], c["end"]))
    return {"metadata": meta, "nodes": nodes, "chunks": chunks}


def split_document(text: str, limit: int, overlap: int) -> list[dict[str, Any]]:
    """Skip blank windows; preserve exact source text and coordinates for all others."""
    if not text:
        return []
    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    pieces = []
    offset = 0
    section = "раздел документа"
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        match = BOUNDARY.match(line)
        is_boundary = bool(match)
        if match:
            section = match.group("section") or line.strip().lstrip("# ")[:120]
        for start in range(0, len(line), limit):
            pieces.append(
                (
                    offset + start,
                    offset + min(start + limit, len(line)),
                    number,
                    section,
                    is_boundary and start == 0,
                )
            )
        offset += len(line)
    chunks = []
    pos = 0
    while pos < len(pieces):
        end = pos + 1
        while end < len(pieces) and not pieces[end][4] and pieces[end][1] - pieces[pos][0] <= limit:
            end += 1
        start_char, end_char = pieces[pos][0], pieces[end - 1][1]
        chunk_text = text[start_char:end_char]
        if chunk_text.strip():
            chunks.append(
                {
                    "start": start_char,
                    "end": end_char,
                    "line_start": bisect_right(line_starts, start_char),
                    "line_end": bisect_right(line_starts, end_char - 1),
                    "section": pieces[pos][3],
                    "text": chunk_text,
                }
            )
        next_pos = end
        if end < len(pieces) and not pieces[end][4]:
            next_pos = max(pos + 1, end - overlap)
        pos = next_pos
    return chunks
