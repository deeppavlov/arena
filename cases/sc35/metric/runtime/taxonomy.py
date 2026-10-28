"""Frozen category proposal authored by the assistant; no per-answer clustering."""

from __future__ import annotations

from projection import digest

CATEGORIES = {
    "sections_content": "Разделы и обязательные сведения",
    "metadata": "Реквизиты и обозначения",
    "annotation_contents": "Аннотация и содержание",
    "page_layout": "Параметры страниц",
    "headings": "Заголовки",
    "paragraphs_lists": "Абзацы, шрифты и списки",
    "figures": "Рисунки",
    "formulas": "Формулы",
    "tables": "Таблицы",
    "notes_footnotes": "Примечания и сноски",
    "references_terms": "Ссылки, термины и сокращения",
    "appendices": "Приложения",
}


def classify_espd(rule_id):
    if rule_id.startswith("ESPD-19.301-"):
        return "sections_content"
    if rule_id.startswith(("ESPD-19.101-", "ESPD-19.103-", "ESPD-19.104-")):
        return "metadata"
    if rule_id.startswith("ESPD-19.105-"):
        return "metadata" if "title_sheet" in rule_id else "annotation_contents"
    if not rule_id.startswith("ESPD-19.106-"):
        raise ValueError("Unknown ESPD rule: " + rule_id)
    for needles, category in (
        (("page_format", "margin_"), "page_layout"),
        (("annotation_numbered", "contents_numbered"), "annotation_contents"),
        (("heading_", "section_centered", "section_uppercase", "subsection_"), "headings"),
        (("paragraph_indent", "list_sort"), "paragraphs_lists"),
        (("figure_",), "figures"),
        (("formula_", "parameter_dimension"), "formulas"),
        (("table_",), "tables"),
        (("note_", "footnote_"), "notes_footnotes"),
        (("term_list", "abbreviation", "reference", "standard_designation"), "references_terms"),
        (("appendix_",), "appendices"),
    ):
        if any(needle in rule_id for needle in needles):
            return category
    raise ValueError("Unclassified ESPD rule: " + rule_id)


def make_taxonomy(rules):
    value = {
        "schema_version": "sc35-span-taxonomy-v1",
        "version": "1.0.0",
        "authorship": (
            "Categories proposed by the LLM assistant during development; "
            "no external classification API call."
        ),
        "categories": CATEGORIES,
        "rule_categories": {r["rule_id"]: classify_espd(r["rule_id"]) for r in rules},
        "freeze_policy": (
            "Fixed before candidate evaluation; changes require a new version and rerun."
        ),
    }
    return {**value, "taxonomy_sha256": digest(value)}


def validate_taxonomy(taxonomy):
    expected = taxonomy.get("taxonomy_sha256")
    actual = digest({k: v for k, v in taxonomy.items() if k != "taxonomy_sha256"})
    if expected != actual or taxonomy.get("schema_version") != "sc35-span-taxonomy-v1":
        raise ValueError("Taxonomy identity mismatch")
    if not set(taxonomy["rule_categories"].values()) <= set(taxonomy["categories"]):
        raise ValueError("Unknown mapped category")
    return taxonomy
