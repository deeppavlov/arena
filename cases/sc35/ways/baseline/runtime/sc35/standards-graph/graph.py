"""Inspect references, plan scoped checks and execute partial DOCX checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent
REFERENCE = re.compile(
    r"ГОСТ\s*(Р\s*)?(\d+(?:[ \t]*\.[ \t]*\d+)*)"
    r"(?:\s*[-‐‑‒–—−]\s*(\d{4}|\d{2})(?!\d))?(?!\d)"
)
KINDS = {
    "normative",
    "informative",
    "foreword",
    "edition_note",
    "profile_selection",
    "internal_dependency",
}


def references(text):
    """Return mentions only: a reference is not proof of applicability."""
    normalized = text.replace("\u00ad", "").replace("\x00", "")
    result = []
    for match in REFERENCE.finditer(normalized):
        number = re.sub(r"\s+", "", match[2])
        designation = "ГОСТ " + ("Р " if match[1] else "") + number
        if match[3]:
            designation += "-" + match[3]
        result.append({"designation": designation, "offset": match.start()})
    return result


def extract(path):
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return [
            {"pdf_page": i + 1, "references": references(page.extract_text() or "")}
            for i, page in enumerate(PdfReader(path).pages)
        ]
    if path.suffix.lower() == ".docx":
        with ZipFile(path) as package:
            root = ElementTree.fromstring(package.read("word/document.xml"))
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        return [
            {
                "all_paragraph_index": i,
                "references": references("".join(t.text or "" for t in p.iter(namespace + "t"))),
            }
            for i, p in enumerate(root.iter(namespace + "p"))
        ]
    raise ValueError("Only PDF and DOCX inputs are supported")


def overlaps(left, right):
    """Clause prefixes match at dot boundaries, so 4.5 does not select 4.50."""
    return any(
        a == "*" or b == "*" or a == b or a.startswith(b + ".") or b.startswith(a + ".")
        for a in left
        for b in right
    )


def validate(catalog):
    nodes = catalog["nodes"]
    index = {node["id"]: node for node in nodes}
    if len(index) != len(nodes):
        raise ValueError("Duplicate node ID")
    edge_ids = set()
    for edge in catalog["edges"]:
        if edge["id"] in edge_ids:
            raise ValueError("Duplicate edge ID")
        edge_ids.add(edge["id"])
        if edge["from"] not in index or edge["to"] not in index:
            raise ValueError("Unknown edge endpoint")
        if edge["kind"] not in KINDS:
            raise ValueError("Unknown edge kind")
        if not edge["source_scope"] or not edge["target_scope"]:
            raise ValueError("Explicit source and target scope are required")
        if edge["review_status"] != "source_checked":
            raise ValueError("Unreviewed proposals belong in the inventory, not active edges")
        source = index[edge["from"]]
        if not source.get("sha256"):
            raise ValueError("Source-checked edge requires a pinned source")
        page = edge["pdf_page"]
        if type(page) is not int or not 1 <= page <= source["page_count"]:
            raise ValueError("Invalid PDF page")
        if edge["kind"] == "internal_dependency" and edge["from"] != edge["to"]:
            raise ValueError("An internal dependency must stay within the same edition")
        if edge["kind"] not in {"profile_selection", "internal_dependency"}:
            evidence = catalog["inventory"][source["id"]]
            mentions = {
                r["designation"]
                for item in evidence
                if item["pdf_page"] == page
                for r in item["references"]
            }
            if edge["literal_reference"] not in mentions:
                raise ValueError("Edge reference is absent from its declared source page")
        for predicate in edge["requires"]:
            if not isinstance(predicate, str) or not predicate:
                raise ValueError("Invalid applicability predicate")
    for profile in catalog["profiles"].values():
        if not profile["roots"] or not set(profile["roots"]) <= set(index):
            raise ValueError("Unknown profile root")
    return index


def verify_sources(catalog, cache):
    validate(catalog)
    count = 0
    for node in catalog["nodes"]:
        if not node.get("sha256"):
            continue
        path = cache / (node["id"] + ".pdf")
        if hashlib.sha256(path.read_bytes()).hexdigest() != node["sha256"]:
            raise ValueError("Source hash differs: " + node["id"])
        observed = extract(path)
        if len(observed) != node["page_count"] or observed != catalog["inventory"][node["id"]]:
            raise ValueError("Reference extraction differs: " + node["id"])
        count += 1
    return {"verified_sources": count, "inventory_reproduced": True}


def plan(catalog, profile_name, facts=None):
    """Cycle-safe traversal; unresolved conditions and missing checks stay visible."""
    nodes = validate(catalog)
    profile = catalog["profiles"][profile_name]
    context = dict(profile["facts"])
    for name, value in (facts or {}).items():
        if value is not None and type(value) is not bool:
            raise ValueError("Facts must be boolean or null")
        context[name] = value
    queue = deque((node_id, ("*",)) for node_id in profile["roots"])
    visited, selected, deferred, skipped = set(), [], [], []
    excluded = {"foreword", "informative", "edition_note"}
    while queue:
        node_id, scope = queue.popleft()
        state = node_id, scope
        if state in visited:
            continue
        visited.add(state)
        node = nodes[node_id]
        bindings = [
            item for item in node.get("local_predicates", []) if overlaps(scope, [item["clause"]])
        ]
        selected.append(
            {
                "node": node_id,
                "scope": list(scope),
                "local_predicate_ids": sorted({item["rule_id"] for item in bindings}),
                "checker_status": "local_dataset_audits_only" if bindings else "not_implemented",
                "reference_inventory_complete": node.get("references_reviewed", False),
                "edition_resolution": node.get("edition_resolution", "not_certified_current"),
            }
        )
        for edge in catalog["edges"]:
            if edge["from"] != node_id or not overlaps(scope, edge["source_scope"]):
                continue
            if edge["kind"] in excluded:
                skipped.append({"edge": edge["id"], "reason": edge["kind"]})
                continue
            conditions = [context.get(key) for key in edge["requires"]]
            if False in conditions:
                skipped.append({"edge": edge["id"], "reason": "condition_false"})
            elif None in conditions:
                deferred.append(
                    {
                        "edge": edge["id"],
                        "missing_facts": [
                            key for key in edge["requires"] if context.get(key) is None
                        ],
                    }
                )
            else:
                queue.append((edge["to"], tuple(sorted(set(edge["target_scope"])))))
    return {
        "profile": profile_name,
        "facts": context,
        "selected": selected,
        "deferred": deferred,
        "skipped": skipped,
        "coverage_complete": False,
        "document_checked": False,
        "unreviewed_mentions": [
            item
            for item in catalog.get("pending_mentions", [])
            if any(item["source"] == state[0] for state in visited)
        ],
        "note": (
            "Plan only. Local corpus audits are not full document checkers or a compliance result."
        ),
    }


def mermaid(catalog):
    validate(catalog)
    aliases = {node["id"]: f"n{i}" for i, node in enumerate(catalog["nodes"])}
    lines = ["flowchart TD"]
    for node in catalog["nodes"]:
        label = node["designation"] + " / " + node.get("edition", "не раскрыт")
        lines.append(f'  {aliases[node["id"]]}["{label}"]')
    for edge in catalog["edges"]:
        style = "-.->" if edge["kind"] in {"foreword", "informative", "edition_note"} else "-->"
        label = edge["clause"] + ": " + edge["condition"]
        lines.append(f'  {aliases[edge["from"]]} {style}|"{label}"| {aliases[edge["to"]]}')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["validate", "verify-sources", "extract", "plan", "check", "mermaid"]
    )
    parser.add_argument("--catalog", type=Path, default=BASE / "catalog.json")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--profile", choices=["pmi", "nir"], default="pmi")
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--review-output", type=Path, help="Write the standalone espd-review-v1")
    args = parser.parse_args()
    for output in (args.output, args.review_output):
        if output and args.input and output.resolve() == args.input.resolve():
            parser.error("Output must not overwrite the input document")
        if output and output.resolve() == args.catalog.resolve():
            parser.error("Output must not overwrite the catalog")
    if args.review_output and args.action != "check":
        parser.error("--review-output requires check")
    if args.review_output and args.output and args.review_output.resolve() == args.output.resolve():
        parser.error("Report and review need different output paths")
    if args.action == "extract":
        if args.input is None:
            parser.error("extract requires --input")
        result = {"input": str(args.input), "mentions_only": True, "locations": extract(args.input)}
    else:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        if args.action == "validate":
            result = {"nodes": len(validate(catalog)), "edges": len(catalog["edges"])}
        elif args.action == "verify-sources":
            if args.cache is None:
                parser.error("verify-sources requires --cache")
            result = verify_sources(catalog, args.cache)
        elif args.action == "mermaid":
            result = mermaid(catalog)
        elif args.action == "check":
            from document_checks import check

            if args.input is None or args.input.suffix.lower() != ".docx":
                parser.error("check requires --input with a DOCX document")
            facts = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts else None
            result = check(
                catalog,
                lambda facts: plan(catalog, args.profile, facts),
                args.input,
                args.case_id,
                facts,
            )
            if args.review_output and result["status"] != "error":
                args.review_output.parent.mkdir(parents=True, exist_ok=True)
                args.review_output.write_text(
                    json.dumps(result["review"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        else:
            facts = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts else None
            result = plan(catalog, args.profile, facts)
    payload = (
        result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if isinstance(result, dict) and result.get("status") == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
