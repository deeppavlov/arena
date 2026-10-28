"""Preserve category/character corpus aggregation under the actual arena protocol."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
from metric import aggregate  # noqa: E402


def combine(payload):
    rows = payload["results"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Cannot aggregate an empty or malformed result set")
    results, seen = [], set()
    for row in rows:
        result = row["statistics"]["span_result"]
        if (
            row["example_id"] != result["case_id"]
            or row["example_id"] in seen
            or result["status"] != "scored"
        ):
            raise ValueError("Incomplete, duplicate or inconsistent metric result")
        seen.add(row["example_id"])
        results.append(result)
    result = aggregate(results)
    return {
        "score": result["score_percent"],
        "statistics": {
            "case_count": len(results),
            "by_category": result["by_category"],
            "micro": result["micro"],
        },
    }


if __name__ == "__main__":
    try:
        json.dump(combine(json.load(sys.stdin)), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    except Exception as error:
        sys.stderr.write(f"SC35 aggregation failed ({type(error).__name__})\n")
        raise SystemExit(1) from None
