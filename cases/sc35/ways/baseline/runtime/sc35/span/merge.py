"""Merge stage outputs already expressed in the same corrupted-text contract."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_dataset import save


def merge(reviews):
    if not reviews:
        raise ValueError("No stage outputs")
    result = copy.deepcopy(reviews[0])
    identities = (
        "schema_version",
        "case_id",
        "document_sha256",
        "text_sha256",
        "taxonomy_sha256",
        "extraction_version",
        "offset_unit",
    )
    result.update(findings=[], unresolved=[], status="ready", stages=[])
    seen = set()
    profiles = {r.get("source_profile", "unspecified") for r in reviews} - {"unspecified"}
    if len(profiles) > 1:
        raise ValueError("Cannot silently merge incompatible normative profiles")
    result["source_profile"] = next(iter(profiles), "unspecified")
    for index, review in enumerate(reviews):
        if any(review.get(key) != reviews[0].get(key) for key in identities):
            raise ValueError("Stages refer to different documents, texts or taxonomies")
        if review.get("status") != "ready" or review.get("unresolved"):
            raise ValueError("A failed/unmapped stage cannot become a successful merged review")
        result["stages"].append(review.get("adapter", f"stage-{index + 1}"))
        for finding in review["findings"]:
            signature = json.dumps(
                {k: v for k, v in finding.items() if k not in {"id", "source_id"}},
                sort_keys=True,
                ensure_ascii=False,
            )
            if signature in seen:
                continue
            seen.add(signature)
            result["findings"].append(
                {
                    **finding,
                    "id": f"F{len(result['findings']) + 1:04d}",
                    "source_stage": index + 1,
                    "source_id": finding["id"],
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in args.reviews}:
        parser.error("Merged output must not overwrite stage inputs")
    result = merge([json.loads(p.read_text(encoding="utf-8")) for p in args.reviews])
    save(args.output, result)


if __name__ == "__main__":
    main()
