"""One baseline with graph, typography and an injectable semantic LLM stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / "typography"))
sys.path.insert(0, str(ROOT / "span"))
from adapters import adapt  # noqa: E402
from build_dataset import save  # noqa: E402
from content_review import review as review_content  # noqa: E402
from content_review import semantic_rules  # noqa: E402
from format_review import check as check_format  # noqa: E402
from format_review import coverage as rule_coverage  # noqa: E402
from merge import merge  # noqa: E402
from projection import Projection, digest  # noqa: E402
from run_graph import load_module  # noqa: E402
from taxonomy import validate_taxonomy  # noqa: E402
from typo_rules import check as check_typography  # noqa: E402
from typo_rules import profile as typography_profile  # noqa: E402


def load_assets(dataset):
    dataset = Path(dataset)
    taxonomy = validate_taxonomy(
        json.loads((dataset / "taxonomy.json").read_text(encoding="utf-8"))
    )
    profile = json.loads((dataset / "profile.json").read_text(encoding="utf-8"))
    if (
        profile.get("profile_id") != "espd-pmi-unified-v1"
        or digest(profile) != taxonomy["profile_sha256"]
        or profile.get("typography_policy") != typography_profile()
    ):
        raise ValueError("Unsupported or changed unified policy")
    return taxonomy, profile


def graph_components():
    graph_dir = ROOT / "standards-graph"
    graph = load_module("unified_graph_planner", graph_dir / "graph.py")
    checker = load_module("unified_graph_checks", graph_dir / "document_checks.py")
    catalog = json.loads((graph_dir / "catalog.json").read_text(encoding="utf-8"))
    return graph, checker, catalog


def review_document(source, taxonomy, profile, case_id, semantic_client=None, components=None):
    projection = Projection(source)
    graph, checker, catalog = components or graph_components()
    raw = checker.check(catalog, lambda facts: graph.plan(catalog, "pmi", facts), source, case_id)
    normative = adapt("graph", raw, projection, taxonomy, case_id)
    for finding in normative["findings"]:
        finding["basis_kind"] = "gost_requirement"
    editorial = check_typography(source, taxonomy, case_id)
    editorial["adapter"] = "typography"
    formal = check_format(source, taxonomy, profile, case_id)
    stages = [normative, formal, editorial]
    content = None
    if semantic_client is not None:
        document = projection.to_dict()
        document.update(case_id=case_id, taxonomy_sha256=taxonomy["taxonomy_sha256"])
        document["formal_evidence"] = {
            "assessments": formal["formal_assessments"],
            "inventory": formal["format_inventory"],
        }
        content = review_content(document, taxonomy, profile, semantic_client)
        stages.append(content)
    result = merge(stages)
    result.update(
        policy_profile="espd-pmi-unified-v1",
        llm_called=bool(content and content["semantic_transport"]["llm_called"]),
        semantic_stage_enabled=content is not None,
        limits=list(dict.fromkeys(limit for stage in stages for limit in stage.get("limits", []))),
    )
    if content is not None:
        result["semantic_assessments"] = content["semantic_assessments"]
        result["semantic_transport"] = content["semantic_transport"]
    result["formal_assessments"] = formal["formal_assessments"]
    result["format_inventory"] = formal["format_inventory"]
    result["configuration_candidates"] = formal["configuration_candidates"]
    result["rule_coverage"] = rule_coverage(profile)
    assessments = {r["rule_id"]: r for r in formal["formal_assessments"]}
    for row in raw.get("checks", []):
        assessments[row["rule_id"]] = row
    for rule in semantic_rules(profile):
        if rule["rule_id"] not in assessments:
            assessments[rule["rule_id"]] = {
                "rule_id": rule["rule_id"],
                "status": "not_run",
                "reason": "Semantic model client was not supplied",
            }
    if content is not None:
        for row in content["semantic_assessments"]:
            previous = assessments.get(row["rule_id"])
            combined = dict(row)
            if previous is not None:
                combined["instrumental_assessment"] = previous
                if previous["status"] == "violations_found":
                    combined["model_assessment"] = dict(row)
                    combined["status"] = "violations_found"
            assessments[row["rule_id"]] = combined
    for rule in profile["typography_policy"]["rules"]:
        assessments[rule["rule_id"]] = {
            "rule_id": rule["rule_id"],
            "status": "violations_found"
            if any(f["rule_id"] == rule["rule_id"] for f in editorial["findings"])
            else "no_violation_observed",
            "reason": "Recognized objects under the editorial policy; see policy limits",
        }
    result["rule_assessments"] = [
        assessments.get(
            rule,
            {
                "rule_id": rule,
                "status": "not_run",
                "reason": "No applicable checker execution",
            },
        )
        for rule in taxonomy["rule_categories"]
    ]
    result["coverage_complete"] = False
    return result


def run(dataset, output, split="train", semantic_client=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("Use a fresh output outside the dataset")
    taxonomy, profile = load_assets(dataset)
    solver = json.loads((dataset / "solver" / (split + ".json")).read_text(encoding="utf-8"))
    if solver["taxonomy_sha256"] != taxonomy["taxonomy_sha256"] or not solver["cases"]:
        raise ValueError("Wrong taxonomy or empty split")
    components = graph_components()
    rows = []
    for case in solver["cases"]:
        source = (dataset / case["input"]).resolve()
        if not source.is_relative_to(dataset / "inputs"):
            raise ValueError("Unsafe solver input")
        projection = Projection(source)
        if (
            projection.source_sha256 != case["input_sha256"]
            or projection.text_sha256 != case["text_sha256"]
        ):
            raise ValueError("Input/canonical text mismatch")
        review = review_document(
            source, taxonomy, profile, case["case_id"], semantic_client, components
        )
        save(output / "reviews" / (case["case_id"] + ".json"), review)
        rows.append(
            {
                "case_id": case["case_id"],
                "stages": review["stages"],
                "findings": len(review["findings"]),
                "status": review["status"],
                "llm_called": review["llm_called"],
            }
        )
    result = {
        "schema_version": "sc35-unified-baseline-v1",
        "case_count": len(rows),
        "split": split,
        "ready_count": sum(c["status"] == "ready" for c in rows),
        "cases": rows,
        "baseline_id": "espd-format-typography-content-v3"
        if semantic_client is not None
        else "espd-format-plus-typography-v3",
        "solver_only": True,
        "llm_called": any(row["llm_called"] for row in rows),
        "llm_judge_enabled": False,
        "stages_per_document": ["graph", "format", "typography"]
        + (["semantic-llm"] if semantic_client is not None else []),
    }
    save(output / "run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "public", "private"], default="train")
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
