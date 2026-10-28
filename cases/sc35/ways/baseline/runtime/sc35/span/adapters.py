"""Input-only adapters for the five tested checker formats; never use references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_dataset import save
from lxml import etree
from projection import NS, VERSION, Projection, digest, span_dicts
from taxonomy import validate_taxonomy

BASE = Path(__file__).resolve().parent
KINDS = ("graph", "gost-review", "gost-skills", "gostforge", "gostdoc")


def external_category(code, message):
    if code == "S.01":
        return "sections_content"
    if code == "S.03":
        return "headings"
    if code.startswith(("T.", "L.")):
        return "paragraphs_lists"
    if code.startswith("SC35-"):
        return "formulas" if code in {"SC35-01", "SC35-05"} else "paragraphs_lists"
    for prefix, category in (
        ("G732-4", "sections_content"),
        ("G732-6.2", "headings"),
        ("G732-6.3", "page_layout"),
        ("G732-6.4", "headings"),
        ("G732-6.5", "figures"),
        ("G732-6.6", "tables"),
        ("G732-6.7", "notes_footnotes"),
        ("G732-6.8", "formulas"),
        ("G732-6.9", "references_terms"),
        ("G732-6.13", "annotation_contents"),
        ("G732-6.16", "references_terms"),
        ("G732-6.17", "appendices"),
    ):
        if code.startswith(prefix):
            return category
    text = (code + " " + message).casefold()
    for pattern, category in (
        (r"формул|formula|^where", "formulas"),
        (r"таблиц|table", "tables"),
        (r"рисунк|рис\.|figure", "figures"),
        (r"приложен|appendix", "appendices"),
        (r"сноск|примечан|footnote|note-", "notes_footnotes"),
        (r"оглавлен|содержани|аннотац|toc|abstract", "annotation_contents"),
        (r"поле|margin|секци|страниц|page-", "page_layout"),
        (r"заголов|heading|^title-", "headings"),
        (r"ссылк|литератур|сокращен|аббрев|reference|bibliograph", "references_terms"),
        (r"обозначен|реквизит|designation", "metadata"),
        (r"структурн|^missing", "sections_content"),
        (
            r"шрифт|кегль|абзац|отступ|интервал|font|size-|indent|spacing|^G732-6\.1",
            "paragraphs_lists",
        ),
    ):
        if re.search(pattern, text, re.I):
            return category
    raise ValueError("Category needs an explicit mapping: " + code)


def rows_for(kind, raw):
    if kind == "graph":
        if raw.get("status") in {"error", "not_implemented"}:
            raise ValueError("Graph checker did not execute successfully")
        return raw.get("review", raw)["findings"]
    if kind in {"gost-review", "gost-skills"}:
        if not isinstance(raw.get("issues"), list):
            raise ValueError("Checker output needs an issues array")
        # review_candidates are kept as a limitation, not promoted to confirmed findings.
        return raw["issues"]
    if kind == "gostforge":
        if not isinstance(raw, list):
            raise ValueError("gostforge output must be a JSON array")
        return raw
    messages = [line.strip()[2:] for line in raw.splitlines() if line.strip().startswith("- ")]
    if not messages and not any(token in raw for token in ("соответств", "Соответств")):
        raise ValueError("Unrecognized gostdoc output; cannot treat as clean")
    return [{"message": message} for message in messages]


def document_anchor(projection):
    return [{"start": len(projection.text) - 1, "end": len(projection.text)}]


def locate(kind, row, projection, mapping):
    if kind == "graph":
        return projection.resolve(row["locator"]), "reported_object"
    if kind == "gost-review":
        loc = dict(row.get("locator", {}))
        if not loc.get("xpath") and type(loc.get("paragraph_index")) is int:
            # Native checker uses all w:p, 1-based, documented by its inventory.
            loc.update(
                paragraph_index=loc["paragraph_index"] - 1,
                paragraph_index_scope="all_document_paragraphs",
            )
        return projection.resolve(loc), "reported_object"
    if kind == "gost-skills":
        spans = []
        for place in row.get("places", []):
            if type(place.get("paragraph")) is not int:
                raise ValueError("gost-skills place has no paragraph index")
            loc = {
                "paragraph_index": place["paragraph"] - 1,
                "paragraph_index_scope": "direct_body_paragraphs",
            }
            spans += projection.resolve(loc)
        if spans:
            return spans, "reported_object"
        code = row.get("code", "")
        if code == "missing":
            return document_anchor(projection), "document_missing_object_anchor"
        if code.startswith("page-number"):
            # Footer has no main-text coordinates; explicit document-property anchor.
            return document_anchor(projection), "document_property_anchor"
        raise ValueError("Checker supplied no positions")
    if kind == "gostforge":
        if mapping and mapping.get("schema_version") == "sc35-gostforge-source-map-v1":
            if mapping["document_sha256"] != projection.source_sha256:
                raise ValueError("Native source map belongs to another input")
            native_ids = re.findall(
                r"(?:p-\d+|sec-\d+|t-\d+|fig-\d+|formula-\d+)", row.get("location", "")
            )
            native = mapping["mapping"]
            if native_ids:
                key = native_ids[0]
                if key not in native:
                    raise ValueError("Native parser ID has no unique captured source location")
                return projection.resolve(native[key]), "native_parser_xml_address"
            if (
                row.get("location") in {"document", "bibliography", "page_sections"}
                or row.get("check_code") == "S.01"
            ):
                return document_anchor(projection), "document_property_anchor"
            if row.get("location", "").startswith("page_sections."):
                return projection.property_anchors() or document_anchor(
                    projection
                ), "section_property_anchor"
            raise ValueError("Unrecognized native location")
        match = re.search(r"paragraph\[([^]]+)\]", row.get("location", ""))
        if match and mapping and match[1] in mapping:
            source = mapping[match[1]]
            quote = source.get("text")
            if quote:
                try:
                    return projection.resolve({"text": quote}), "mapped_unique_quote"
                except ValueError:
                    if not source.get("source_body_paragraphs"):
                        raise
            paragraphs = source.get("source_body_paragraphs", [])
            spans = []
            selected = []
            for paragraph in paragraphs:
                if not re.fullmatch(r"P\d+", paragraph):
                    raise ValueError("Unknown gostforge source index")
                direct_index = int(paragraph[1:]) - 1
                if not 0 <= direct_index < len(projection.direct):
                    raise ValueError("Source-map paragraph index does not exist")
                selected.append(projection.direct[direct_index])
                spans += projection.resolve(
                    {
                        "paragraph_index": int(paragraph[1:]) - 1,
                        "paragraph_index_scope": "direct_body_paragraphs",
                    }
                )
            if spans:
                if quote:
                    canonical = [projection.by_element[p]["text"] for p in selected]
                    # The pinned parser omits OMML. Verify its legacy visible text
                    # at the supplied addresses, but return full canonical spans.
                    legacy = ["".join(p.xpath(".//w:t/text()", namespaces=NS)) for p in selected]

                    def normalize(value):
                        return " ".join(value.split())

                    alternatives = [
                        normalize(separator.join(values))
                        for values in (canonical, legacy)
                        for separator in ("", " ")
                    ]
                    if normalize(quote) not in alternatives:
                        raise ValueError("Source-map paragraph and quote disagree")
                return spans, "provided_source_mapping"
        if row.get("source_text") and len(row["source_text"]) == 1:
            return projection.resolve({"text": row["source_text"][0]}), "mapped_unique_quote"
        if not match and any(s in row.get("location", "") for s in ("page", "style")):
            return document_anchor(projection), "document_property_anchor"
        raise ValueError("gostforge requires a source map or an unambiguous source quote")
    message = row["message"]
    section = re.search(r"Секция\s+(\d+)", message)
    if section:
        sections = projection.root.findall(".//w:sectPr", NS)
        index = int(section[1]) - 1
        if not 0 <= index < len(sections):
            raise ValueError("Reported section does not exist")
        return span_dicts(projection.node_spans(sections[index])), "section_property_anchor"
    if any(token in message for token in ("Стиль", "колонтитул", "номера страницы")):
        return document_anchor(projection), "document_property_anchor"
    raise ValueError("Unrecognized gostdoc location")


def adapt(kind, raw, projection, taxonomy, case_id, mapping=None, category_overrides=None):
    if kind not in KINDS:
        raise ValueError("Unknown checker adapter")
    source_hash = (
        raw.get("input_sha256", raw.get("document_sha256")) if isinstance(raw, dict) else None
    )
    if source_hash and source_hash != projection.source_sha256:
        raise ValueError("Raw checker output refers to a different input")
    result = {
        "schema_version": "sc35-span-review-v1",
        "case_id": case_id,
        "document_sha256": projection.source_sha256,
        "text_sha256": projection.text_sha256,
        "extraction_version": VERSION,
        "offset_unit": "unicode_codepoint",
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "source_profile": "espd-pmi" if kind == "graph" else "nir-7.32",
        "adapter": kind,
        "raw_output_sha256": digest(raw),
        "status": "ready",
        "findings": [],
        "unresolved": [],
        "limits": [
            "Adapter converts locations and categories; it does not validate the normative claim.",
            "Raw input hash is checked when present; otherwise input binding is caller-supplied.",
            "Document/property anchors represent non-text errors, not measured page geometry.",
        ],
    }
    try:
        rows = rows_for(kind, raw)
    except (ValueError, KeyError) as error:
        result.update(status="error", unresolved=[{"error": str(error)}])
        return result
    if kind == "gost-review" and raw.get("review_candidates"):
        result["limits"].append("Native review_candidates are not promoted to confirmed issues.")
    for i, row in enumerate(rows):
        code = str(row.get("rule_id", row.get("code", row.get("check_code", ""))))
        message = row.get("description", row.get("message", ""))
        try:
            category = (category_overrides or {}).get(code)
            if category is None:
                category = taxonomy["rule_categories"].get(code) or external_category(code, message)
            if category not in taxonomy["categories"]:
                raise ValueError("Unknown mapped category")
            spans, anchor = locate(kind, row, projection, mapping)
            # One native report issue with several places can contain several instances.
            groups = [[span] for span in spans] if kind == "gost-skills" else [spans]
            for j, group in enumerate(groups):
                result["findings"].append(
                    {
                        "id": f"F{i + 1:04d}-{j + 1}",
                        "category": category,
                        "spans": group,
                        "message": message,
                        "rule_id": code,
                        "anchor_kind": anchor,
                        "source_id": row.get("id", row.get("finding_id", str(i))),
                    }
                )
        except (ValueError, KeyError, TypeError, etree.XPathError) as error:
            result["unresolved"].append(
                {"source_index": i, "code": code, "message": message, "reason": str(error)}
            )
    if result["unresolved"]:
        result["status"] = "needs_mapping"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--categories", type=Path, help="Explicit native-code to category mapping")
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.raw.resolve()}:
        parser.error("Do not overwrite source inputs")
    text = args.raw.read_text(encoding="utf-8-sig")
    raw = text if args.kind == "gostdoc" else json.loads(text)
    taxonomy = validate_taxonomy(json.loads((BASE / "taxonomy.json").read_text(encoding="utf-8")))
    result = adapt(
        args.kind,
        raw,
        Projection(args.input),
        taxonomy,
        args.case_id or args.input.stem,
        json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None,
        json.loads(args.categories.read_text(encoding="utf-8")) if args.categories else None,
    )
    save(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "findings": len(result["findings"]),
                "unresolved": len(result["unresolved"]),
            }
        )
    )
    raise SystemExit(0 if result["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
