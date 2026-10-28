"""Run the existing graph checker and adapt it, using solver-side data only."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from adapters import adapt
from build_dataset import save
from projection import Projection
from taxonomy import validate_taxonomy

BASE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(dataset, output, split="train"):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset or output in dataset.parents or dataset in output.parents:
        raise ValueError("Output must stay outside the dataset")
    if output.exists():
        raise ValueError("Use a fresh output directory to exclude stale submissions")
    taxonomy = validate_taxonomy(
        json.loads((dataset / "taxonomy.json").read_text(encoding="utf-8"))
    )
    solver = json.loads((dataset / "solver" / (split + ".json")).read_text(encoding="utf-8"))
    if not solver["cases"]:
        raise ValueError("Empty split")
    graph_dir = BASE.parent / "standards-graph"
    graph = load_module("span_graph_planner", graph_dir / "graph.py")
    checks = load_module("span_graph_checker", graph_dir / "document_checks.py")
    catalog = json.loads((graph_dir / "catalog.json").read_text(encoding="utf-8"))
    rows = []
    for case in solver["cases"]:
        key = case["case_id"]
        source = (dataset / case["input"]).resolve()
        if not source.is_relative_to(dataset / "inputs"):
            raise ValueError("Unsafe solver input path")
        projection = Projection(source)
        if (
            projection.source_sha256 != case["input_sha256"]
            or projection.text_sha256 != case["text_sha256"]
        ):
            raise ValueError("Solver input/canonical text hashes differ")
        raw = checks.check(catalog, lambda facts: graph.plan(catalog, "pmi", facts), source, key)
        review = adapt("graph", raw, projection, taxonomy, key)
        save(output / "raw" / (key + ".json"), raw)
        save(output / "reviews" / (key + ".json"), review)
        rows.append(
            {
                "case_id": key,
                "status": review["status"],
                "findings": len(review["findings"]),
                "unresolved": len(review["unresolved"]),
            }
        )
    result = {
        "schema_version": "sc35-span-baseline-v1",
        "split": split,
        "case_count": len(rows),
        "ready_count": sum(r["status"] == "ready" for r in rows),
        "finding_count": sum(r["findings"] for r in rows),
        "cases": rows,
        "solver_only": True,
        "llm_called": False,
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
    }
    save(output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "public", "private"], default="train")
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    raise SystemExit(0 if result["ready_count"] == result["case_count"] else 1)


if __name__ == "__main__":
    main()
