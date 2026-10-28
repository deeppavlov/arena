"""Input-only, partial DOCX checks dispatched by a scoped standards graph plan.

No GOLD, corruption annotations, fixture contexts or model responses are read.
The result is evidence for selected rules, never a full compliance certificate.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = {"w": W[1:-1]}
NODE301 = "gost-19.301-79@1987"
NODE106 = "gost-19.106-78@2010"
SECTIONS = (
    ("object", ("Объект испытаний",), "2.1"),
    ("goal", ("Цель испытаний",), "2.2"),
    ("program_requirements", ("Требования к программе",), "2.3"),
    ("documentation_requirements", ("Требования к программной документации",), "2.4"),
    ("means_procedure", ("Средства и порядок испытаний", "Состав и порядок испытаний"), "2.7"),
    ("methods", ("Методы испытаний",), "2.8"),
)
LIMITS = [
    "Only selected rules and pinned editions are implemented; expert approval is pending.",
    "Heading recognition uses body paragraphs, outline inheritance and explicit section names.",
    "Unrecognized heading aliases, images and unsupported OOXML structures require review.",
    "No PDF measurements, pagination, font, margin or semantic sufficiency checks are performed.",
    "Nonempty section content does not establish that its required information is correct.",
    "Headers, footers, tracked changes and automatic line wrapping are not evaluated.",
]


def paragraph_text(element):
    parts = []
    for child in element.iter():
        if any(parent.tag in {W + "del", W + "moveFrom"} for parent in child.iterancestors()):
            continue
        if child.tag == W + "t":
            parts.append(child.text or "")
        elif child.tag == W + "tab":
            parts.append("\t")
        elif child.tag == W + "cr" or (
            child.tag == W + "br" and child.get(W + "type", "textWrapping") == "textWrapping"
        ):
            parts.append("\n")
    return "".join(parts)


def normalized_title(value):
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", value)
    return " ".join(value.strip().rstrip(".").split()).casefold()


class Document:
    def __init__(self, path):
        self.path = Path(path)
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        with ZipFile(self.path) as package:
            self.root = etree.fromstring(package.read("word/document.xml"), parser)
            self.styles = {}
            if "word/styles.xml" in package.namelist():
                styles = etree.fromstring(package.read("word/styles.xml"), parser)
                self.styles = {s.get(W + "styleId"): s for s in styles.findall("w:style", NS)}
        self.body = self.root.find("w:body", NS)
        if self.root.tag != W + "document" or self.body is None:
            raise ValueError("DOCX has no Word document body")
        self.paragraphs = self.body.findall("w:p", NS)
        self.indices = {p: i for i, p in enumerate(self.paragraphs)}
        self.blocks = list(self.body)
        self.block_indices = {block: i for i, block in enumerate(self.blocks)}
        self._chains, self._toc, self._outlines, self._names, self._texts = {}, {}, {}, {}, {}
        self.aliases = {
            normalized_title(alias): key for key, aliases, _ in SECTIONS for alias in aliases
        }
        self.unsupported = bool(
            self.body.xpath(
                ".//w:altChunk | .//w:sdt | .//w:ins | .//w:del | .//w:moveFrom", namespaces=NS
            )
        )
        self.visual_content = bool(
            self.body.xpath(".//w:drawing | .//w:pict | .//w:object", namespaces=NS)
        )

    def text(self, element):
        if element not in self._texts:
            self._texts[element] = paragraph_text(element)
        return self._texts[element]

    def style_chain(self, paragraph):
        if paragraph in self._chains:
            return self._chains[paragraph]
        tag = paragraph.find("w:pPr/w:pStyle", NS)
        current = self.styles.get(tag.get(W + "val")) if tag is not None else None
        seen, chain = set(), []
        while current is not None and current.get(W + "styleId") not in seen:
            seen.add(current.get(W + "styleId"))
            chain.append(current)
            base = current.find("w:basedOn", NS)
            current = self.styles.get(base.get(W + "val")) if base is not None else None
        self._chains[paragraph] = chain
        return chain

    def is_toc(self, paragraph):
        if paragraph in self._toc:
            return self._toc[paragraph]
        for style in self.style_chain(paragraph):
            name = style.find("w:name", NS)
            label = (
                (name.get(W + "val", "") if name is not None else "")
                + " "
                + (style.get(W + "styleId") or "")
            ).casefold()
            if re.search(r"\b(toc\s*\d*|содержание\s*\d+)\b", label):
                self._toc[paragraph] = True
                return True
        self._toc[paragraph] = bool(
            paragraph.xpath(".//w:instrText | .//w:fldSimple | .//w:hyperlink", namespaces=NS)
        )
        return self._toc[paragraph]

    def outline_level(self, paragraph):
        if paragraph in self._outlines:
            return self._outlines[paragraph]
        for item in [paragraph, *self.style_chain(paragraph)]:
            level = item.find("w:pPr/w:outlineLvl", NS)
            if level is not None:
                self._outlines[paragraph] = int(level.get(W + "val", "9"))
                return self._outlines[paragraph]
        self._outlines[paragraph] = None
        return None

    def section_name(self, paragraph):
        if paragraph in self._names:
            return self._names[paragraph]
        if self.is_toc(paragraph):
            return None
        self._names[paragraph] = self.aliases.get(normalized_title(self.text(paragraph)))
        return self._names[paragraph]

    def is_heading(self, paragraph):
        if self.is_toc(paragraph):
            return False
        level = self.outline_level(paragraph)
        return (level is not None and 0 <= level < 9) or self.section_name(paragraph) is not None

    def section_headings(self, key):
        return [p for p in self.paragraphs if self.section_name(p) == key]

    def section_blocks(self, heading):
        start = self.block_indices[heading] + 1
        end = len(self.blocks)
        for block in self.blocks[start:]:
            if block.tag == W + "sectPr":
                end = self.block_indices[block]
                break
            if block.tag != W + "p" or self.is_toc(block):
                continue
            if (
                self.section_name(block) is not None
                or self.outline_level(block) == 0
                or re.match(r"^приложение\b", normalized_title(self.text(block)))
            ):
                end = self.block_indices[block]
                break
        return self.blocks[start:end]

    def locator(self, paragraph=None, title=None):
        result = {"part": "word/document.xml", "xpath": "/w:document/w:body"}
        if paragraph is not None:
            index = self.indices[paragraph]
            result.update(
                xpath=f"/w:document/w:body/w:p[{index + 1}]",
                paragraph_index=index,
                paragraph_index_scope="direct_body_paragraphs",
                text=self.text(paragraph),
            )
        if title:
            result["section"] = title
        return result


def supported_rules():
    result = {}
    for key, _, clause in SECTIONS:
        result[f"ESPD-19.301-79-1.2-section_missing_{key}"] = (NODE301, "presence", key)
        if key != "means_procedure":
            result[f"ESPD-19.301-79-{clause}-empty_section"] = (NODE301, "empty", key)
    for clause, kind in (
        ("2.1.3", "annotation_numbered"),
        ("2.1.4", "contents_numbered"),
        ("2.1.7", "heading_final_period"),
        ("2.1.7", "heading_word_break"),
    ):
        result[f"ESPD-19.106-78-{clause}-{kind}"] = (NODE106, kind, None)
    return result


def check_rule(document, kind, key):
    """Return observations and unresolved reasons; no unobserved property is a pass."""
    if kind in {"presence", "empty"}:
        _, aliases, _ = next(item for item in SECTIONS if item[0] == key)
        headings = document.section_headings(key)
        if len(headings) > 1:
            return [], ["Ambiguous section headings"]
        if not headings:
            if kind == "empty":
                return [], ["Section absent; content cannot be assessed"]
            if document.visual_content or document.unsupported:
                return [], [
                    "Section name may be in unsupported or non-text content; inspect render"
                ]
            return [
                {
                    "description": f"Не найден обязательный раздел «{aliases[0]}».",
                    "locator": document.locator(title=aliases[0]),
                    "evidence": ["В доступных абзацах основной части не найден заголовок раздела."],
                }
            ], []
        if kind == "presence":
            return [], []
        blocks = document.section_blocks(headings[0])
        if any(
            block.tag in {W + "altChunk", W + "sdt", W + "ins", W + "del"}
            or block.xpath(
                ".//w:altChunk | .//w:sdt | .//w:ins | .//w:del | .//w:moveFrom", namespaces=NS
            )
            for block in blocks
        ):
            return [], ["Unsupported structures inside this section require review"]
        if any(document.text(block).strip() for block in blocks):
            return [], []
        if any(
            block.xpath(
                ".//w:drawing | .//w:pict | .//w:object | .//w:fldSimple | .//w:fldChar | "
                ".//*[local-name()='oMath']",
                namespaces=NS,
            )
            for block in blocks
        ):
            return [], ["Non-text or field content requires inspection; section not declared empty"]
        return [
            {
                "description": f"В разделе «{aliases[0]}» отсутствует содержимое.",
                "locator": document.locator(headings[0], aliases[0]),
                "evidence": [
                    "Между заголовком и следующей границей раздела нет текста или объектов."
                ],
            }
        ], []
    observations = []
    for paragraph in document.paragraphs:
        if document.is_toc(paragraph):
            continue
        value = document.text(paragraph)
        matched = False
        if kind == "heading_final_period":
            matched = document.is_heading(paragraph) and value.rstrip().endswith(".")
            description = "Точка в конце заголовка."
        elif kind == "heading_word_break":
            matched = document.is_heading(paragraph) and bool(
                re.search(r"[^\W\d_]-\n[^\W\d_]", value)
            )
            description = "Явный перенос слова внутри заголовка."
        else:
            title = "Аннотация" if kind == "annotation_numbered" else "Содержание"
            matched = bool(re.fullmatch(r"\s*\d+[.)]?\s+" + title + r"\s*\.?\s*", value, re.I))
            if document.outline_level(paragraph) is None and document.unsupported:
                matched = False
            description = f"«{title}» пронумеровано как раздел."
        if matched:
            observations.append(
                {
                    "description": description,
                    "locator": document.locator(paragraph),
                    "evidence": [value],
                }
            )
    limitations = (
        ["Unsupported body structures are outside this text scan"] if document.unsupported else []
    )
    return observations, limitations


def execute(catalog, planned, document, case_id):
    nodes = {node["id"]: node for node in catalog["nodes"]}
    registry = supported_rules()
    findings, checks, errors, states = [], [], [], []
    seen = set()
    for selected in planned["selected"]:
        node = nodes[selected["node"]]
        executed = []
        for rule_id in selected["local_predicate_ids"]:
            binding = registry.get(rule_id)
            if binding is None or binding[0] != node["id"]:
                continue
            executed.append(rule_id)
            if rule_id in seen:
                continue
            seen.add(rule_id)
            clause = next(r["clause"] for r in node["local_predicates"] if r["rule_id"] == rule_id)
            try:
                observed, reasons = check_rule(document, binding[1], binding[2])
                for finding in observed:
                    findings.append(
                        {
                            **finding,
                            "id": f"F{len(findings) + 1:04d}",
                            "rule_id": rule_id,
                            "normative_basis": [
                                {
                                    "standard": node["designation"],
                                    "clause": clause,
                                    "source": node["source_url"],
                                    "edition_node": node["id"],
                                }
                            ],
                        }
                    )
                status = (
                    "needs_review"
                    if reasons
                    else ("violations_found" if observed else "no_violation_observed")
                )
                checks.append({"rule_id": rule_id, "status": status, "limitations": reasons})
            except (ValueError, KeyError, TypeError) as error:
                errors.append({"rule_id": rule_id, "error": str(error)})
                checks.append({"rule_id": rule_id, "status": "error"})
        states.append(
            {
                **selected,
                "executed_rule_ids": executed,
                "unsupported_rule_ids": sorted(
                    set(selected["local_predicate_ids"]) - set(executed)
                ),
                "runtime_status": "partially_checked" if executed else "not_implemented",
            }
        )
    return {
        "schema_version": "sc35-graph-review-v1",
        "profile": planned["profile"],
        "input": str(document.path),
        "input_sha256": document.sha256,
        "status": "error" if errors else ("partial" if checks else "not_implemented"),
        "document_checked": bool(checks),
        "coverage_complete": False,
        "expert_approved": False,
        "review": {"schema_version": "espd-review-v1", "case_id": case_id, "findings": findings},
        "checks": checks,
        "errors": errors,
        "selected": states,
        "facts": planned["facts"],
        "deferred": planned["deferred"],
        "skipped": planned["skipped"],
        "unreviewed_mentions": planned["unreviewed_mentions"],
        "limits": LIMITS,
    }


def check(catalog, planner, path, case_id=None, facts=None):
    if any(value is not None and type(value) is not bool for value in (facts or {}).values()):
        raise ValueError("Facts must be boolean or null")
    document = Document(path)
    observed = {"has_tables": bool(document.body.findall(".//w:tbl", NS))}
    for key, value in observed.items():
        if (facts or {}).get(key) is not None and facts[key] != value:
            raise ValueError("Supplied fact contradicts DOCX: " + key)
    planned = planner({**(facts or {}), **observed})
    return execute(catalog, planned, document, case_id or Path(path).stem)
