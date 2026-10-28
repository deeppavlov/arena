"""Opt-in editorial policies for PMI; these are not additional GOST clauses."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

SPAN = Path(__file__).resolve().parents[1] / "span"
sys.path.insert(0, str(SPAN))
from projection import NS, VERSION, Projection, W  # noqa: E402

RULES = [
    ("TYP-01", "paragraphs_lists", "Моноширинный шрифт технического блока", [1]),
    ("TYP-02", "tables", "Отсутствие абзацных отступов в таблице", [2]),
    ("TYP-03", "tables", "Без выключки по ширине в таблице", [2]),
    ("TYP-04", "tables", "Одинарный автоматический интервал в таблице", [2]),
    ("TYP-05", "tables", "Вертикальное выравнивание ячейки по верху", [2]),
    ("TYP-06", "tables", "Текстовая ячейка выровнена влево", [2]),
    ("TYP-07", "paragraphs_lists", "Грамматическое тире в обычном тексте", [3]),
    ("TYP-08", "paragraphs_lists", "Неразрывный пробел после выбранных коротких слов", [4, 8]),
    ("TYP-09", "headings", "Нет конечной точки в заголовке", [6]),
    ("TYP-10", "tables", "Нет конечной точки в однословной ячейке-фрагменте", [6]),
    ("TYP-11", "figures", "Нет конечной точки в подписи рисунка", [6]),
    ("TYP-12", "headings", "Нет скачка уровня заголовка вниз", [14]),
    ("TYP-13", "headings", "Нет явной отмены нумерации нумерованного заголовка", [15]),
    ("TYP-14", "paragraphs_lists", "Шрифт текста из согласованного семейства", [10]),
]
MONO = {"courier new", "consolas", "liberation mono", "dejavu sans mono", "noto sans mono"}
BODY_FONTS = {
    "times new roman",
    "arial",
    "liberation serif",
    "liberation sans",
    "nimbus roman",
    "nimbus sans",
}


def profile():
    return {
        "profile_id": "espd-pmi-typography-editorial-v1",
        "version": "1.0.0",
        "basis_kind": "editorial_recommendation",
        "expert_approved": False,
        "source": "docs/studenikin/2026-10-27_SC-35_типографика.md",
        "book_catalogue": "https://www.artlebedev.ru/izdal/spravochnik-izdatelya-i-avtora-2017/",
        "book_clause_verification": "pending_full_text_and_page_references",
        "rules": [
            {
                "rule_id": key,
                "category": category,
                "title": title,
                "excerpt_items": items,
                "basis_kind": "editorial_recommendation",
                "severity": "recommendation",
            }
            for key, category, title, items in RULES
        ],
        "limits": [
            "Opt-in local policy, not a claim of mandatory GOST or full handbook compliance.",
            "Short-word spacing is a prevention proxy, not proof of a rendered hanging word.",
            "Numeric alignment, line fitting, table pagination and semantic editing are deferred.",
            "No automatic slash ban; code, URLs and paths require their original characters.",
            "Explicit code and text-cell styles establish applicability for narrow policies.",
            "Unknown theme fonts and table-style inheritance are not inferred.",
        ],
    }


class Properties:
    def __init__(self, path):
        with ZipFile(path) as package:
            raw = (
                package.read("word/styles.xml") if "word/styles.xml" in package.namelist() else None
            )
        self.root = (
            etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            if raw
            else None
        )
        self.styles = (
            {}
            if self.root is None
            else {n.get(W + "styleId"): n for n in self.root.findall("w:style", NS)}
        )
        self.default = next(
            (
                n.get(W + "styleId")
                for n in self.styles.values()
                if n.get(W + "type") == "paragraph" and n.get(W + "default") == "1"
            ),
            None,
        )

    def style_chain(self, identifier):
        seen = set()
        while identifier in self.styles and identifier not in seen:
            seen.add(identifier)
            style = self.styles[identifier]
            yield style
            parent = style.find("w:basedOn", NS)
            identifier = parent.get(W + "val") if parent is not None else None

    def style_id(self, paragraph):
        node = paragraph.find("w:pPr/w:pStyle", NS)
        return node.get(W + "val") if node is not None else self.default

    def containers(self, paragraph, run=None):
        kind = "rPr" if run is not None else "pPr"
        if run is not None:
            yield run.find("w:rPr", NS)
            identifier = run.find("w:rPr/w:rStyle", NS)
            if identifier is not None:
                for style in self.style_chain(identifier.get(W + "val")):
                    yield style.find("w:rPr", NS)
        else:
            yield paragraph.find("w:pPr", NS)
        for style in self.style_chain(self.style_id(paragraph)):
            yield style.find("w:" + kind, NS)
        if self.root is not None:
            yield self.root.find(f"w:docDefaults/w:{kind}Default/w:{kind}", NS)

    def attr(self, paragraph, element, attribute="val", run=None):
        for container in self.containers(paragraph, run):
            node = container.find("w:" + element, NS) if container is not None else None
            if element == "rFonts" and node is not None and node.get(W + attribute + "Theme"):
                return None  # Theme resolution requires a separate font/rendering model.
            if node is not None and node.get(W + attribute) is not None:
                return node.get(W + attribute)
        return None

    def level(self, paragraph):
        level = self.attr(paragraph, "outlineLvl")
        if level is not None and level.isdigit() and int(level) < 9:
            return int(level) + 1
        match = re.fullmatch(r"Heading([1-9])", self.style_id(paragraph) or "", re.I)
        return int(match[1]) if match else None


def check(path, taxonomy, case_id):
    projection, properties = Projection(path), Properties(path)
    findings, previous_level = [], None
    titles = {key: title for key, _, title, _ in RULES}

    def add(rule, paragraph, local=None):
        start, end = projection.paragraph_span(paragraph)
        if local is not None:
            start = projection.by_element[paragraph]["start"] + local[0]
            end = projection.by_element[paragraph]["start"] + local[1]
        findings.append(
            {
                "id": f"T{len(findings) + 1:04d}",
                "rule_id": rule,
                "category": taxonomy["rule_categories"][rule],
                "spans": [{"start": start, "end": end}],
                "message": titles[rule],
                "anchor_kind": "text_object" if local else "object",
                "basis_kind": "editorial_recommendation",
                "severity": "recommendation",
            }
        )

    for paragraph in projection.paragraphs:
        text = projection.by_element[paragraph]["text"]
        style = properties.style_id(paragraph) or ""
        code = style.lower() in {"sc35code", "code", "sourcecode"}
        cell = next((n for n in paragraph.iterancestors() if n.tag == W + "tc"), None)
        level = properties.level(paragraph)
        if not text.strip():
            continue
        if code:
            fonts = [
                properties.attr(paragraph, "rFonts", "ascii", run=r)
                or properties.attr(paragraph, "rFonts", "hAnsi", run=r)
                for r in paragraph.findall("w:r", NS)
                if r.find("w:t", NS) is not None
            ]
            if any(font and font.lower() not in MONO for font in fonts):
                add("TYP-01", paragraph)
            continue
        # A whole technical token is not ordinary prose.
        technical = bool(
            re.fullmatch(
                r"(?:https?://\S+|[A-Za-z]:[\\/].*|/[\w/.-]+|[\w.-]+/[\w/.-]+)", text.strip()
            )
        )
        if not technical:
            for match in re.finditer(r"(?<=[А-Яа-яЁё]) ([–-]) (?=[А-Яа-яЁё])", text):
                add("TYP-07", paragraph, match.span(1))
            for match in re.finditer(r"(?<![\w/\\])([АвВИикКоОсСуУа]) (?=[А-Яа-яЁё])", text):
                add("TYP-08", paragraph, match.span())
        if level:
            if text.rstrip().endswith(".") and not text.rstrip().endswith("..."):
                end = len(text.rstrip())
                add("TYP-09", paragraph, (end - 1, end))
            if previous_level is not None and level > previous_level + 1:
                add("TYP-12", paragraph)
            previous_level = level
            direct_num = paragraph.find("w:pPr/w:numPr/w:numId", NS)
            inherited = [s.find("w:pPr/w:numPr/w:numId", NS) for s in properties.style_chain(style)]
            inherited = next((n.get(W + "val") for n in inherited if n is not None), None)
            if (
                direct_num is not None
                and direct_num.get(W + "val") == "0"
                and inherited not in {None, "0"}
            ):
                add("TYP-13", paragraph)
        caption = bool(re.match(r"^Рисунок\s+\d+\s+[—–-]\s+", text))
        if caption and text.rstrip().endswith(".") and not text.rstrip().endswith("..."):
            end = len(text.rstrip())
            add("TYP-11", paragraph, (end - 1, end))
        if cell is not None:
            leading = properties.attr(paragraph, "ind", "start")
            if leading is None:
                leading = properties.attr(paragraph, "ind", "left")
            indent = [
                leading,
                properties.attr(paragraph, "ind", "firstLine"),
                properties.attr(paragraph, "ind", "hanging"),
            ]
            if any(value is not None and int(value) != 0 for value in indent):
                add("TYP-02", paragraph)
            alignment = properties.attr(paragraph, "jc")
            if alignment in {"both", "distribute"}:
                add("TYP-03", paragraph)
            line, line_rule = (
                properties.attr(paragraph, "spacing", "line"),
                properties.attr(paragraph, "spacing", "lineRule"),
            )
            if line and line_rule in {None, "auto"} and int(line) > 240:
                add("TYP-04", paragraph)
            vertical = cell.find("w:tcPr/w:vAlign", NS)
            first = next(cell.iter(W + "p"), None)
            if paragraph is first and vertical is not None and vertical.get(W + "val") != "top":
                add("TYP-05", paragraph)
            if style == "SC35TableText" and alignment in {"right", "center", "end"}:
                add("TYP-06", paragraph)
            word = text.strip()
            if (
                re.fullmatch(r"[А-Яа-яЁё][А-Яа-яЁё-]+\.", word)
                and word[:-1].lower() not in {"см", "мм", "табл", "рис", "стр"}
                and not word[:-1].isupper()
            ):
                end = len(text.rstrip())
                add("TYP-10", paragraph, (end - 1, end))
        fonts = [
            properties.attr(paragraph, "rFonts", "ascii", run=r)
            or properties.attr(paragraph, "rFonts", "hAnsi", run=r)
            for r in paragraph.findall("w:r", NS)
            if r.find("w:t", NS) is not None
        ]
        if any(font and font.lower() not in BODY_FONTS for font in fonts):
            add("TYP-14", paragraph)
    return {
        "schema_version": "sc35-span-review-v1",
        "case_id": case_id,
        "document_sha256": projection.source_sha256,
        "text_sha256": projection.text_sha256,
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "extraction_version": VERSION,
        "offset_unit": "unicode_codepoint",
        "source_profile": "espd-pmi",
        "policy_profile": "espd-pmi-typography-editorial-v1",
        "status": "ready",
        "findings": findings,
        "unresolved": [],
        "basis_kind": "editorial_recommendation",
        "limits": profile()["limits"],
    }
