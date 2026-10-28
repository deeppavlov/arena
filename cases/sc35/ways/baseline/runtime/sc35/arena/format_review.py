"""Input-only formal ESPD checks and explicit per-rule coverage.

This module never imports mutation builders, reads reference labels or identifies
fixture bookmarks. OOXML configuration findings are marked partial; they do not
certify the position of printed objects. Absence of a drawn border is permitted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "span"))
sys.path.insert(0, str(BASE.parent / "typography"))
from content_review import semantic_rules  # noqa: E402
from projection import NS, VERSION, Projection, W, span_dicts  # noqa: E402
from typo_rules import Properties  # noqa: E402

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
MARGIN_MM = {"left": 20, "right": 10, "top": 25, "bottom": 15}
GRAPH_SLUGS = {
    "annotation_numbered",
    "contents_numbered",
    "heading_final_period",
    "heading_word_break",
    "empty_section",
}
LAYOUT = {
    "page_format",
    "margin_bottom",
    "margin_left",
    "margin_right",
    "margin_top",
    "section_centered",
    "subsection_indent",
    "paragraph_indent",
    "heading_after_spacing",
    "heading_before_spacing",
    "heading_line_spacing",
    "appendix_page_break",
    "appendix_label_position",
    "appendix_title_center",
    "formula_number_position",
    "figure_number_position",
    "figure_title_position",
    "figure_explanation_position",
    "table_footnote_position",
    "footnote_marker",
}
TEXT = {
    "section_uppercase",
    "subsection_case",
    "heading_number_sequence",
    "heading_number_style",
    "formula_number_missing",
    "formula_number_sequence",
    "formula_part_prefix",
    "formula_reference",
    "formula_symbol_newline",
    "formula_where_colon",
    "formula_where_missing",
    "figure_number_missing",
    "figure_number_sequence",
    "figure_reference",
    "figure_repeated_reference",
    "table_title_case",
    "single_note_label",
    "single_note_numbered",
    "multiple_note_label",
    "multiple_note_numbering",
    "appendix_number",
    "appendix_title_case",
    "appendix_reference_missing",
    "appendix_toc_missing",
}
METADATA = {
    "designation_wrong_pmi_kind",
    "designation_country_separator",
    "designation_edition_separator",
    "designation_kind_separator",
    "designation_organization_separator",
    "designation_part_separator",
    "designation_registration_separator",
    "designation_kind_group_missing",
    "designation_edition_zero",
    "designation_number_zero",
    "designation_part_zero",
    "approval_designation_suffix",
    "title_designation_missing",
    "title_program_title_case",
    "title_year_suffix",
    "title_sheet_missing",
    "contents_entry_address_missing",
    "contents_entry_number_missing",
    "contents_entry_title_missing",
    "means_procedure_empty",
}


def slug(rule):
    return rule["rule_id"].split("-", 4)[-1]


def coverage(profile):
    """Routes describe implemented capability, never document compliance."""
    result = []
    model_ids = {r["rule_id"] for r in semantic_rules(profile)}
    editorial_ids = {r["rule_id"] for r in profile.get("typography_policy", {}).get("rules", [])}
    for rule in profile["rules"]:
        key = slug(rule)
        if rule["rule_id"] in editorial_ids:
            backend, level = "typography", "editorial_recognized_objects_only"
        elif rule.get("evidence_channel") == "semantic":
            backend, level = "semantic-llm", "requires_model"
        elif key in GRAPH_SLUGS or key.startswith("section_missing_"):
            backend, level = "graph", "text_structure"
        elif key in LAYOUT:
            backend, level = "format", "configuration_only_requires_render"
        elif key in TEXT or key in METADATA:
            backend, level = "format", "recognized_objects_only"
        else:
            backend, level = "none", "not_implemented"
        result.append(
            {
                "rule_id": rule["rule_id"],
                "backend": backend,
                "coverage": level,
                "content_review_enabled": rule["rule_id"] in model_ids,
            }
        )
    result.extend(
        {
            "rule_id": r["rule_id"],
            "backend": "typography",
            "coverage": "editorial_recognized_objects_only",
        }
        for r in profile.get("typography_policy", {}).get("rules", [])
        if r["rule_id"] not in {item["rule_id"] for item in result}
    )
    return result


def integer(value):
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class Formal:
    def __init__(self, path, taxonomy, profile, case_id):
        self.p = Projection(path)
        self.props = Properties(path)
        self.taxonomy, self.profile = taxonomy, profile
        self.rules = {slug(r): r for r in profile["rules"] if r.get("evidence_channel") == "ooxml"}
        self.ps = self.p.paragraphs
        self.body = self.p.root.find("w:body", NS)
        self.direct = list(self.body.findall("w:p", NS))
        self.findings, self.checks = [], {}
        self.candidates = []
        self.result = {
            "schema_version": "sc35-span-review-v1",
            "case_id": case_id,
            "document_sha256": self.p.source_sha256,
            "text_sha256": self.p.text_sha256,
            "taxonomy_sha256": taxonomy["taxonomy_sha256"],
            "extraction_version": VERSION,
            "offset_unit": "unicode_codepoint",
            "source_profile": "espd-pmi",
            "adapter": "format",
            "status": "ready",
            "unresolved": [],
            "findings": self.findings,
        }
        self.headings = [p for p in self.direct if self.level(p) is not None and not self.is_toc(p)]
        self.sections = [p for p in self.headings if self.level(p) == 1]
        self.formulas = []
        for p in self.ps:
            equations = p.findall(".//" + M + "oMath")
            if not equations:
                continue
            prose = self.text(p)
            for equation in equations:
                math_text = "".join(equation.itertext())
                prose = prose.replace(math_text, "", 1)
            # Inline symbols/equations within prose are not numbered displays.
            if p.find(".//" + M + "oMathPara") is not None or re.fullmatch(
                r"\s*(?:\([\d.]+\))?\s*", prose
            ):
                self.formulas.append(p)
        self.appendices = [
            p
            for p in self.direct
            if re.fullmatch(r"ПРИЛОЖЕНИЕ(?:\s+[^\n]+)?", self.text(p).strip())
        ]
        self.toc = [p for p in self.ps if self.is_toc(p)]
        for key in self.rules:
            self.state(key, "not_applicable", "No recognized applicable object")

    def text(self, p):
        return self.p.by_element[p]["text"]

    def is_toc(self, p):
        style = self.props.style_id(p) or ""
        return bool(
            re.match(r"(?:TOC|Contents|Содержание)\s*\d", style, re.I)
            or p.xpath(".//w:instrText[contains(., 'TOC')]", namespaces=NS)
        )

    def level(self, p):
        value = self.props.level(p)
        if value is not None:
            return value
        text = self.text(p).strip()
        if re.match(r"^\d+\s+[А-ЯЁA-Z][А-ЯЁA-Z\s\d]+$", text):
            return 1
        return None

    def shown(self, p):
        chunks = []
        for run in p.findall(".//w:r", NS):
            value = "".join(run.xpath("w:t/text()", namespaces=NS))
            caps = False
            for container in reversed(list(self.props.containers(p, run))):
                node = container.find("w:caps", NS) if container is not None else None
                if node is not None:
                    caps = node.get(W + "val", "true") not in {"0", "false", "off"}
            if caps:
                value = value.upper()
            chunks.append(value)
        return "".join(chunks) or self.text(p)

    def state(self, key, status, reason):
        if key in self.rules:
            priority = {
                "not_applicable": 0,
                "no_violation_observed": 1,
                "needs_context": 2,
                "needs_render": 3,
                "violations_found": 4,
            }
            previous = self.checks.get(key, {}).get("status", "not_applicable")
            if priority.get(previous, 0) > priority.get(status, 0):
                return
            self.checks[key] = {
                "rule_id": self.rules[key]["rule_id"],
                "status": status,
                "reason": reason,
            }

    def checked(self, key):
        if self.checks.get(key, {}).get("status") != "violations_found":
            self.state(key, "no_violation_observed", "Checked recognized objects only")

    def add(self, key, node, message, partial=False):
        if key not in self.rules:
            return
        rule = self.rules[key]
        target = self.candidates if partial else self.findings
        target.append(
            {
                "id": f"C{len(target) + 1:04d}" if partial else f"O{len(target) + 1:04d}",
                "rule_id": rule["rule_id"],
                "category": self.taxonomy["rule_categories"][rule["rule_id"]],
                "spans": span_dicts(self.p.node_spans(node)),
                "message": message,
                "anchor_kind": "configuration_anchor" if partial else "reported_object",
                "basis_kind": "gost_requirement",
                "coverage_level": "partial" if partial else "recognized_object",
                "evidence_channel": "ooxml",
                "requires_render": partial,
                "normative_basis": [
                    {
                        "standard": rule["standard"],
                        "clause": rule["clause"],
                        "source": rule.get("source"),
                    }
                ],
            }
        )
        self.state(
            key,
            "needs_render" if partial else "violations_found",
            "Configuration discrepancy requires render confirmation"
            if partial
            else "Violation in a recognized object",
        )

    def ambiguous_position(self, p):
        value = self.text(p)
        return bool(
            value[:1].isspace()
            or "\t" in value
            or p.xpath(".//w:drawing | .//w:pict | .//w:framePr", namespaces=NS)
            or any(n.tag in {W + "tc", W + "txbxContent"} for n in p.iterancestors())
        )

    def alignment(self, key, p, allowed):
        value = self.props.attr(p, "jc")
        if self.ambiguous_position(p) or value is None:
            self.state(
                key, "needs_render", "Alternative positioning or unknown effective alignment"
            )
        elif value not in allowed:
            self.add(
                key,
                p,
                f"Выравнивание OOXML: {value}; ожидается {', '.join(sorted(allowed))}. "
                "Подтвердить расположение при рендере.",
                True,
            )
        else:
            self.checked(key)

    def layout(self):
        for section in self.p.root.findall(".//w:sectPr", NS):
            size = section.find("w:pgSz", NS)
            width = integer(size.get(W + "w")) if size is not None else None
            height = integer(size.get(W + "h")) if size is not None else None
            if width is None or height is None:
                self.state("page_format", "needs_context", "Page dimensions are unavailable")
            elif not any(
                abs(min(width, height) - x) <= 5 and abs(max(width, height) - y) <= 5
                for x, y in ((11906, 16838), (16838, 23811))
            ):
                self.add(
                    "page_format",
                    section,
                    "Размер страницы OOXML отличается от A4/A3. "
                    "Нужны сведения о способе изготовления и допуске устройства.",
                    True,
                )
            else:
                self.checked("page_format")
            # Margins alone do not define a text field: paragraph indents, tables,
            # text frames, mirror margins and gutters may implement the same form.
            margins = section.find("w:pgMar", NS)
            for side, mm in MARGIN_MM.items():
                value = integer(margins.get(W + side)) if margins is not None else None
                key = "margin_" + side
                if value is None:
                    self.state(key, "needs_context", "No explicit page margin")
                elif abs(value - mm * 1440 / 25.4) > 5:
                    self.state(
                        key,
                        "needs_render",
                        f"OOXML margin {side}={value} twips differs from form field {mm}mm; "
                        "equivalent layout must be checked",
                    )
                else:
                    self.checked(key)
        for p in self.headings:
            text = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", self.shown(p)).strip()
            if self.level(p) == 1:
                self.checked("section_uppercase")
                if any(c.isalpha() for c in text) and text != text.upper():
                    self.add(
                        "section_uppercase", p, "Заголовок раздела записан не прописными буквами."
                    )
                self.alignment("section_centered", p, {"center"})
            else:
                self.state(
                    "subsection_case",
                    "needs_context",
                    "Machine-produced headings permit printer-font alternatives; "
                    "proper nouns and abbreviations require context",
                )
                self.indent("subsection_indent", p)
            if re.match(r"^[IVXLCDM]+\s", self.text(p)):
                self.add(
                    "heading_number_style",
                    p,
                    "В номере структурного элемента использованы римские цифры вместо арабских.",
                )
            else:
                self.checked("heading_number_style")
            for key in ("heading_before_spacing", "heading_after_spacing", "heading_line_spacing"):
                self.state(
                    key,
                    "needs_render",
                    "Distances depend on manufacturing method, font metrics, "
                    "blank lines and page breaks",
                )
        for p in self.direct:
            if (
                re.match(r"^\d+(?:\.\d+){1,3}\s+", self.text(p))
                and p not in self.headings
                and not self.is_toc(p)
            ):
                self.indent("paragraph_indent", p)
        # Consecutive recognized headings: do not compare an appendix or a
        # separately numbered document part with the main section sequence.
        for previous, current in zip(self.sections, self.sections[1:], strict=False):
            a, b = (
                re.match(r"^(\d+)\s", self.text(previous)),
                re.match(r"^(\d+)\s", self.text(current)),
            )
            if a and b and self.direct.index(current) - self.direct.index(previous) < 30:
                self.checked("heading_number_sequence")
                if int(b[1]) != int(a[1]) + 1:
                    self.add(
                        "heading_number_sequence",
                        current,
                        "Нарушена последовательность номеров соседних разделов; "
                        "проверить границы частей документа.",
                        True,
                    )

    def indent(self, key, p):
        first = integer(self.props.attr(p, "ind", "firstLine"))
        chars = integer(self.props.attr(p, "ind", "firstLineChars"))
        hanging = integer(self.props.attr(p, "ind", "hanging"))
        if (
            self.ambiguous_position(p)
            or chars is not None
            or any(
                c is not None and c.find("w:numPr", NS) is not None
                for c in self.props.containers(p)
            )
        ):
            self.state(
                key,
                "needs_render",
                "Character, list or manual positioning requires layout resolution",
            )
        elif first is None:
            self.state(key, "needs_context", "Effective first-line indent is unknown")
        elif first <= 0 or hanging is not None and hanging > 0:
            self.add(
                key,
                p,
                "У структурного элемента нет положительного абзацного отступа OOXML; "
                "проверить эквивалентное размещение.",
                True,
            )
        else:
            self.checked(key)

    def notes(self):
        for i, p in enumerate(self.ps):
            value = self.text(p).strip()
            if re.match(r"^Примечание\b", value, re.I):
                self.checked("single_note_label")
                self.checked("single_note_numbered")
                if not re.match(r"^Примечание\.", value, re.I):
                    self.add("single_note_label", p, "После слова «Примечание» требуется точка.")
                if re.match(r"^Примечание\s*[.:]?\s*\d+[.)]", value, re.I):
                    self.add("single_note_numbered", p, "Единственное примечание пронумеровано.")
            elif re.match(r"^Примечания\b", value, re.I):
                self.checked("multiple_note_label")
                if not re.match(r"^Примечания:", value, re.I):
                    self.add(
                        "multiple_note_label", p, "После слова «Примечания» требуется двоеточие."
                    )
                rows = []
                for q in self.ps[i + 1 :]:
                    match = re.match(r"^(\d+)([.)])\s", self.text(q).strip())
                    if not match:
                        break
                    rows.append((q, match))
                if rows:
                    self.checked("multiple_note_numbering")
                for index, (q, match) in enumerate(rows, 1):
                    if int(match[1]) != index or match[2] != ".":
                        self.add(
                            "multiple_note_numbering",
                            q,
                            "Нумерация примечаний должна быть последовательной, "
                            "арабскими цифрами с точкой.",
                        )

    def equations(self):
        numbered = []
        for p in self.formulas:
            value = self.text(p)
            match = re.search(r"\((\d+(?:\.\d+)?)\)\s*$", value)
            if len(self.formulas) > 1:
                self.checked("formula_number_missing")
                if match is None:
                    self.add(
                        "formula_number_missing",
                        p,
                        "При нескольких формулах отсутствует распознанный арабский номер "
                        "в скобках; номер в отдельной ячейке требует проверки.",
                        True,
                    )
            if match:
                numbered.append((p, match[1]))
                self.alignment("formula_number_position", p, {"right", "end"})
            index = self.ps.index(p)
            following = []
            for q in self.ps[index + 1 : index + 12]:
                if q in self.formulas or q in self.headings:
                    break
                if re.match(
                    r"^(?:где\s*:?[ \t]*)?[A-Za-zА-Яа-яα-ωΑ-Ω]\s*[—–-]", self.text(q).strip()
                ):
                    following.append(q)
                elif following or self.text(q).strip():
                    break
            if following:
                first = following[0]
                self.checked("formula_where_missing")
                self.checked("formula_where_colon")
                if not re.match(r"^где\b", self.text(first).strip(), re.I):
                    self.add(
                        "formula_where_missing",
                        first,
                        "Первая строка распознанной расшифровки формулы "
                        "не начинается со слова «где».",
                    )
                elif re.match(r"^где\s*:", self.text(first).strip(), re.I):
                    self.add(
                        "formula_where_colon",
                        first,
                        "После слова «где» в расшифровке формулы стоит двоеточие.",
                    )
                for q in following:
                    self.checked("formula_symbol_newline")
                    for line in self.text(q).splitlines():
                        if (
                            len(
                                re.findall(
                                    r"(?:^|;\s*)(?:где\s*:?\s*)?[A-Za-zА-Яа-я]\s*[—–-]", line
                                )
                            )
                            > 1
                        ):
                            self.add(
                                "formula_symbol_newline",
                                q,
                                "Несколько пояснений символов находятся на одной строке.",
                            )
            else:
                for key in (
                    "formula_where_missing",
                    "formula_where_colon",
                    "formula_symbol_newline",
                ):
                    self.state(
                        key,
                        "needs_context",
                        "No recognized definition block; "
                        "a global symbol list is a valid alternative",
                    )
        if len(numbered) > 1:
            self.checked("formula_number_sequence")
            previous = None
            for p, number in numbered:
                if "." in number:
                    self.state(
                        "formula_number_sequence",
                        "needs_context",
                        "Multipart numbering requires part boundaries",
                    )
                    continue
                if previous is not None and int(number) != previous + 1:
                    self.add(
                        "formula_number_sequence",
                        p,
                        "Нарушена последовательность распознанных номеров формул.",
                    )
                previous = int(number)
        if self.formulas:
            self.state(
                "formula_part_prefix",
                "needs_context",
                "The number and boundaries of the current document part are not provided",
            )
            # Inline references can be validated without guessing which number
            # should be cited: require parentheses only for an explicit citation.
            for p in self.ps:
                if p not in self.formulas and re.search(
                    r"\bформул[аеуы]\s+\d+\b", self.text(p), re.I
                ):
                    self.add(
                        "formula_reference",
                        p,
                        "В явной ссылке на формулу номер записан без круглых скобок.",
                    )

    def illustrations(self):
        captions = []
        for p in self.ps:
            match = re.match(r"^\s*(?:Рис\.|Рисунок)\s*(\d+)\b", self.text(p), re.I)
            if match and not self.is_toc(p):
                captions.append((p, int(match[1])))
        pictures = [p for p in self.ps if p.xpath(".//w:drawing | .//w:pict", namespaces=NS)]
        if pictures:
            self.state(
                "figure_number_missing",
                "needs_context",
                "Images can be logos, formulas or parts of a single illustration; "
                "caption count alone is inconclusive",
            )
            for key in (
                "figure_number_position",
                "figure_title_position",
                "figure_explanation_position",
            ):
                self.state(
                    key,
                    "needs_render",
                    "Floating and inline illustrations require rendered position mapping",
                )
        if len(captions) > 1:
            for (_previous, a), (current, b) in zip(captions, captions[1:], strict=False):
                self.checked("figure_number_sequence")
                if b != a + 1:
                    self.add(
                        "figure_number_sequence",
                        current,
                        "Нарушена последовательность распознанных номеров иллюстраций; "
                        "проверить границы приложений.",
                        True,
                    )
        references = {}
        for p in self.ps:
            if any(p is q for q, _ in captions) or self.is_toc(p):
                continue
            for match in re.finditer(r"\bрис\.\s*(\d+)\b", self.text(p), re.I):
                number = int(match[1])
                self.checked("figure_reference")
                self.checked("figure_repeated_reference")
                if number in references and not re.search(
                    r"см\.\s*$", self.text(p)[: match.start()], re.I
                ):
                    self.add(
                        "figure_repeated_reference",
                        p,
                        "Повторная распознанная ссылка на рисунок не содержит «см.».",
                    )
                references[number] = True
        if captions:
            self.state(
                "figure_reference",
                "needs_context",
                "Absence of a recognized citation is not proof that no equivalent reference exists",
            )

    def appendices_and_tables(self):
        for p in self.appendices:
            self.checked("appendix_number")
            if len(self.appendices) > 1 and not re.fullmatch(
                r"ПРИЛОЖЕНИЕ\s+[1-9]\d*", self.text(p).strip()
            ):
                self.add(
                    "appendix_number",
                    p,
                    "При нескольких приложениях требуется арабский номер без знака №.",
                )
            self.alignment("appendix_label_position", p, {"right", "end"})
            self.state(
                "appendix_page_break",
                "needs_render",
                "Automatic pagination, section starts and explicit breaks are valid alternatives",
            )
            index = self.direct.index(p)
            title = next((q for q in self.direct[index + 1 :] if self.text(q).strip()), None)
            if title is not None:
                self.checked("appendix_title_case")
                if self.shown(title) != self.shown(title).upper():
                    self.add(
                        "appendix_title_case",
                        title,
                        "Тематический заголовок приложения записан не прописными буквами.",
                    )
                self.alignment("appendix_title_center", title, {"center"})
            for key in ("appendix_reference_missing", "appendix_toc_missing"):
                self.state(
                    key,
                    "needs_context",
                    "Need semantic reference matching, optional PMI contents "
                    "and separate-document applicability",
                )
        for p in self.ps:
            value = self.text(p).strip()
            if re.match(r"^Таблица\s+\d+\s*[—–-]\s*\S", value, re.I):
                self.checked("table_title_case")
                title = re.sub(r"^Таблица\s+\d+\s*[—–-]\s*", "", self.shown(p), flags=re.I)
                # Entirely uppercase titles may be legitimate abbreviations.
                if title and title[0].islower():
                    self.add(
                        "table_title_case", p, "Заголовок таблицы начинается со строчной буквы."
                    )
        if self.body.findall(".//w:tbl", NS):
            self.state(
                "table_footnote_position",
                "needs_render",
                "Table continuation and notes need pagination, including repeated table headers",
            )
        self.state(
            "footnote_marker",
            "needs_context",
            "Footnote relationships and automatic numbering require dedicated mapping",
        )

    def metadata(self):
        # Explicit designations only. Do not treat references to standards or
        # dotted software versions as designations of the document under review.
        designation = re.compile(
            r"(?<!\w)(?P<country>[A-ZА-Я]{2})(?P<country_sep>[.:/])"
            r"(?P<organization>[A-ZА-Яa-zа-я\d]+)(?P<organization_sep>[.:/])"
            r"(?P<registration>\d{5})(?P<registration_sep>[-/])"
            r"(?P<edition>\d{2})(?P<edition_sep>[ .])"
            r"(?P<kind>\d{2})(?P<kind_sep>[ -])(?P<number>\d{2})"
            r"(?:(?P<part_sep>[-/])(?P<part>\d{1,2}))?(?P<approval>-ЛУ)?(?!\w)"
        )
        expected = {
            "country": ("designation_country_separator", "."),
            "organization": ("designation_organization_separator", "."),
            "registration": ("designation_registration_separator", "-"),
            "edition": ("designation_edition_separator", " "),
            "kind": ("designation_kind_separator", " "),
            "part": ("designation_part_separator", "-"),
        }
        for p in self.ps:
            value = self.text(p)
            for match in designation.finditer(value):
                groups = match.groupdict()
                for component, (key, separator) in expected.items():
                    actual = groups[component + "_sep"]
                    if actual is not None:
                        self.checked(key)
                        if actual != separator:
                            self.add(
                                key,
                                p,
                                f"Разделитель группы {component}: "
                                f"{actual!r}; по ГОСТ требуется {separator!r}.",
                            )
                for key in (
                    "designation_edition_zero",
                    "designation_number_zero",
                    "designation_part_zero",
                    "designation_wrong_pmi_kind",
                ):
                    self.checked(key)
                if groups["edition"] == "00":
                    self.add(
                        "designation_edition_zero",
                        p,
                        "Номер издания в обозначении равен 00 вместо диапазона 01–99.",
                    )
                if groups["number"] == "00":
                    self.add(
                        "designation_number_zero",
                        p,
                        "Номер документа данного вида равен 00 вместо диапазона 01–99.",
                    )
                if groups["part"] is not None and not 1 <= int(groups["part"]) <= 9:
                    self.add("designation_part_zero", p, "Номер части в обозначении равен 0.")
                if groups["kind"] != "51" and re.search(
                    r"обозначени[ея]\s+(?:документа\s*)?[:：]", value, re.I
                ):
                    self.add(
                        "designation_wrong_pmi_kind",
                        p,
                        "В явно указанном обозначении ПМИ код вида отличается от 51; "
                        "проверить выбранную редакцию ГОСТ 19.101.",
                    )
            index = self.ps.index(p)
            previous = self.ps[index - 1] if index else None
            preceding = self.text(previous).strip() if previous is not None else ""
            if preceding.casefold() in {"утвержден", "утверждено"}:
                parsed = designation.fullmatch(value.strip())
                if parsed:
                    self.checked("approval_designation_suffix")
                    if parsed["approval"] is None:
                        self.add(
                            "approval_designation_suffix",
                            p,
                            "В обозначении распознанного листа утверждения отсутствует -ЛУ.",
                        )
            if re.fullmatch(r"Наименование программы\s*[:：]?", preceding, re.I):
                self.checked("title_program_title_case")
                if any(c.isalpha() for c in value) and self.shown(p) != self.shown(p).upper():
                    self.add(
                        "title_program_title_case",
                        p,
                        "В явно обозначенном поле названия программы есть строчные буквы.",
                    )
            cover = bool(self.sections) and index < self.ps.index(self.sections[0])
            if cover and re.fullmatch(r"\s*\d{4}\s*(?:г\.?|год(?:а)?)\s*", value, re.I):
                self.add(
                    "title_year_suffix",
                    p,
                    "Год в распознанной титульной части записан со словом или сокращением.",
                )
            if re.search(r"\bОбозначение\s+документа\s*[:：]\s*$", value, re.I):
                # A standalone label may be followed by a value paragraph or a
                # separate table cell. Unknown geometry is not proof of absence.
                self.state(
                    "title_designation_missing",
                    "needs_context",
                    "An explicit label has no inline value; inspect following paragraph/cell",
                )
            if re.fullmatch(r"[A-ZА-Я]{2}\.[A-ZА-Яa-zа-я\d]+\.\d{5}-\d{2} \d{2}", value.strip()):
                nearby = " ".join(self.text(q) for q in self.ps[max(0, index - 4) : index + 5])
                if re.search("программа и методика испытаний", nearby, re.I):
                    self.add(
                        "designation_kind_group_missing",
                        p,
                        "В распознанном обозначении ПМИ нет отдельных групп вида и номера.",
                    )
        for key in METADATA:
            if (
                self.checks.get(key, {}).get("status") == "not_applicable"
                and key != "means_procedure_empty"
            ):
                self.state(
                    key,
                    "needs_context",
                    "Object roles and equivalent title/TOC/designation representations "
                    "need contextual identification",
                )
        # Empty means/procedure is the one structural rule not executed by the
        # original graph checker. Other section checks remain in that checker.
        for i, p in enumerate(self.direct):
            if re.fullmatch(
                r"(?:\d+[.)]?\s+)?(?:Средства|Состав) и порядок испытаний\.?",
                self.text(p).strip(),
                re.I,
            ):
                following = []
                for q in self.direct[i + 1 :]:
                    if self.level(q) == 1:
                        break
                    following.append(q)
                if not any(
                    self.text(q).strip() or q.xpath(".//w:drawing | .//w:pict", namespaces=NS)
                    for q in following
                ):
                    # A table is valid content; direct paragraphs alone are not
                    # enough to establish an empty section.
                    siblings = list(self.body)
                    start = siblings.index(p) + 1
                    end = siblings.index(
                        next((q for q in self.direct[i + 1 :] if self.level(q) == 1), siblings[-1])
                    )
                    if not any(n.tag == W + "tbl" for n in siblings[start:end]):
                        self.add(
                            "means_procedure_empty",
                            p,
                            "Распознанный раздел средств и порядка испытаний пуст.",
                        )
                else:
                    self.checked("means_procedure_empty")

    def inventory(self, path):
        # Public evidence about all styles/parts, including regions that cannot
        # yet be represented by main-text spans. Never invent a GOST font family.
        sizes, fonts, indents = set(), set(), set()
        unknown_fonts = unknown_sizes = 0
        for p in self.ps:
            indents.add(
                tuple(
                    self.props.attr(p, "ind", a) for a in ("left", "right", "firstLine", "hanging")
                )
            )
            for r in p.findall(".//w:r", NS):
                size = self.props.attr(p, "sz", run=r)
                font = self.props.attr(p, "rFonts", "ascii", run=r) or self.props.attr(
                    p, "rFonts", "hAnsi", run=r
                )
                if size is not None:
                    sizes.add(size)
                else:
                    unknown_sizes += 1
                if font is not None:
                    fonts.add(font)
                else:
                    unknown_fonts += 1
        styles = []
        for identifier, node in self.props.styles.items():
            chain, seen, current = [], set(), identifier
            while current in self.props.styles and current not in seen:
                chain.append(current)
                seen.add(current)
                parent = self.props.styles[current].find("w:basedOn", NS)
                current = parent.get(W + "val") if parent is not None else None
            styles.append(
                {
                    "id": identifier,
                    "type": node.get(W + "type"),
                    "based_on_chain": chain,
                    "cycle": current in seen,
                    "missing_parent": current is not None and current not in self.props.styles,
                }
            )
        parts = []
        with ZipFile(path) as package:
            for name in package.namelist():
                if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name):
                    root = etree.fromstring(
                        package.read(name), etree.XMLParser(resolve_entities=False, no_network=True)
                    )
                    parts.append(
                        {
                            "part": name,
                            "text_characters": sum(
                                len(t) for t in root.xpath(".//w:t/text()", namespaces=NS)
                            ),
                            "tables": len(root.findall(".//w:tbl", NS)),
                            "drawings": len(root.xpath(".//w:drawing | .//w:pict", namespaces=NS)),
                        }
                    )
        return {
            "effective_font_families": sorted(fonts),
            "effective_font_sizes_half_points": sorted(sizes),
            "paragraph_indent_configurations": len(indents),
            "styles": len(self.props.styles),
            "style_inventory": styles,
            "unresolved_font_runs": unknown_fonts,
            "unresolved_size_runs": unknown_sizes,
            "headers_footers": parts,
            "page_borders": len(self.p.root.findall(".//w:pgBorders", NS)),
            "border_absence_is_violation": False,
            "requires_render": [
                "actual_text_field",
                "title_block_geometry",
                "frame_geometry",
                "floating_shapes",
                "automatic_pagination",
            ],
        }


def check(path, taxonomy, profile, case_id):
    review = Formal(path, taxonomy, profile, case_id)
    review.layout()
    review.notes()
    review.equations()
    review.illustrations()
    review.appendices_and_tables()
    review.metadata()
    review.result["formal_assessments"] = [
        review.checks[slug(r)]
        for r in profile["rules"]
        if r.get("evidence_channel") == "ooxml"
        and slug(r) not in GRAPH_SLUGS
        and not slug(r).startswith("section_missing_")
    ]
    review.result["format_inventory"] = review.inventory(path)
    review.result["configuration_candidates"] = review.candidates
    review.result["limits"] = [
        "Input-only recognized OOXML objects; "
        "configuration discrepancies require rendering confirmation.",
        "No fixed TNR/14pt rule is invented; "
        "unknown styles and alternative renderings are not compliance passes.",
        "Frames may be absent; image/table/line title blocks require geometry verification.",
    ]
    return review.result
