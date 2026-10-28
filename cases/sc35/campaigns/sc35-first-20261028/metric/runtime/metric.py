"""Deterministic category/character overlap, plus one-to-one event diagnostics."""

from __future__ import annotations

from projection import VERSION, intervals

METRIC_VERSION = "sc35-span-overlap-v1"


def length(spans):
    return sum(end - start for start, end in intervals(spans))


def intersection(left, right):
    left, right = intervals(left), intervals(right)
    total, i, j = 0, 0, 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def iou(left, right):
    overlap = intersection(left, right)
    union = length(left) + length(right) - overlap
    return overlap / union if union else 0.0


def ratios(overlap, predicted, gold):
    return {
        "precision": overlap / predicted if predicted else None,
        "recall": overlap / gold if gold else None,
        "f1": 2 * overlap / (predicted + gold) if predicted + gold else 1.0,
        "iou": overlap / (predicted + gold - overlap) if predicted + gold - overlap else 1.0,
        "precision_percent": round(100 * overlap / predicted, 6) if predicted else None,
        "recall_percent": round(100 * overlap / gold, 6) if gold else None,
    }


def validate_findings(findings, text_length, categories):
    if not isinstance(findings, list):
        raise ValueError("findings must be an array")
    seen = set()
    parsed = []
    for finding in findings:
        key = finding.get("id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Finding IDs must be unique nonempty strings")
        seen.add(key)
        if finding.get("category") not in categories:
            raise ValueError("Unknown category")
        if not isinstance(finding.get("message"), str) or not finding["message"].strip():
            raise ValueError("Each finding needs a message")
        spans = finding.get("spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError("Each finding must include at least one span")
        values = []
        for span in spans:
            start, end = span.get("start"), span.get("end")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= text_length
            ):
                raise ValueError("Invalid Unicode half-open span")
            values.append((start, end))
        parsed.append({**finding, "intervals": intervals(values)})
    return parsed


def event_matching(gold, predicted, threshold):
    """Maximum cardinality matching; an event can be credited at most once."""
    edges = {
        g: sorted(
            (p for p in range(len(predicted)) if iou(gold[g], predicted[p]) >= threshold),
            key=lambda p: (-iou(gold[g], predicted[p]), p),
        )
        for g in range(len(gold))
    }
    assigned = {}

    def visit(gold_index, seen):
        for pred_index in edges[gold_index]:
            if pred_index in seen:
                continue
            seen.add(pred_index)
            if pred_index not in assigned or visit(assigned[pred_index], seen):
                assigned[pred_index] = gold_index
                return True
        return False

    for g in range(len(gold)):
        visit(g, set())
    return len(assigned)


def evaluate(reference, review, taxonomy, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("Event IoU threshold must be in (0,1]")
    if reference.get("schema_version") != "sc35-span-reference-v1":
        raise ValueError("Invalid span reference")
    if review.get("schema_version") != "sc35-span-review-v1":
        raise ValueError("Invalid span review")
    for key in ("case_id", "document_sha256", "text_sha256", "taxonomy_sha256"):
        if reference.get(key) != review.get(key) or not reference.get(key):
            raise ValueError("Review identity mismatch: " + key)
    if reference["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]:
        raise ValueError("Wrong frozen taxonomy")
    if (
        review.get("extraction_version") != VERSION
        or reference.get("extraction_version") != VERSION
    ):
        raise ValueError("Wrong canonical text version")
    if (
        review.get("offset_unit") != "unicode_codepoint"
        or reference.get("offset_unit") != "unicode_codepoint"
    ):
        raise ValueError("Offsets must be Unicode code points")
    if review.get("status") != "ready" or review.get("unresolved"):
        raise ValueError("Review has unmapped findings or execution errors; not an empty success")
    if review.get("source_profile", "espd-pmi") not in {"espd-pmi", "unspecified"}:
        raise ValueError("Checker profile is not compatible with the ESPD corpus")
    size = reference["text_length"]
    gold = validate_findings(reference["findings"], size, taxonomy["categories"])
    predictions = validate_findings(review["findings"], size, taxonomy["categories"])
    scope = reference["evaluation_scope"]
    regions = [(s["start"], s["end"]) for s in scope["regions"]]
    eligible, out_of_scope, duplicate_count = [], [], 0
    seen = set()
    for finding in predictions:
        key = finding.get("rule_id")
        if (
            key in taxonomy["rule_categories"]
            and taxonomy["rule_categories"][key] != finding["category"]
        ):
            raise ValueError("Category contradicts supplied rule_id")
        if (
            key in taxonomy["rule_categories"] and key not in scope["rule_ids"]
        ) or not intersection(finding["intervals"], regions):
            out_of_scope.append(finding["id"])
            continue
        signature = finding["category"], tuple(finding["intervals"])
        if signature in seen:
            duplicate_count += 1
        else:
            seen.add(signature)
            eligible.append(finding)
    rows = {}
    for category in taxonomy["categories"]:
        gs = [f["intervals"] for f in gold if f["category"] == category]
        ps = [f["intervals"] for f in eligible if f["category"] == category]
        # Parent categories cannot distinguish two causes with identical spans.
        gs = [list(s) for s in dict.fromkeys(tuple(s) for s in gs)]
        if not gs and not ps:
            continue
        gold_union = intervals([span for finding in gs for span in finding])
        pred_union = intervals([span for finding in ps for span in finding])
        common, psize, gsize = (
            intersection(gold_union, pred_union),
            length(pred_union),
            length(gold_union),
        )
        tp = event_matching(gs, ps, threshold)
        rows[category] = {
            "intersection_chars": common,
            "predicted_chars": psize,
            "gold_chars": gsize,
            **ratios(common, psize, gsize),
            "events": {"tp": tp, "fp": len(ps) - tp, "fn": len(gs) - tp},
        }
    common = sum(r["intersection_chars"] for r in rows.values())
    psize = sum(r["predicted_chars"] for r in rows.values())
    gsize = sum(r["gold_chars"] for r in rows.values())
    return {
        "schema_version": "sc35-span-score-v1",
        "metric_version": METRIC_VERSION,
        "case_id": reference["case_id"],
        "status": "scored",
        "assessment_kind": "controlled_mutation_category_localization",
        "by_category": rows,
        "micro": {
            "intersection_chars": common,
            "predicted_chars": psize,
            "gold_chars": gsize,
            **ratios(common, psize, gsize),
        },
        "macro_f1": sum(r["f1"] for r in rows.values()) / len(rows) if rows else 1.0,
        "out_of_scope_ids": out_of_scope,
        "duplicates": duplicate_count,
        "event_iou_threshold": threshold,
        "normative_correctness_verified": False,
    }


def aggregate(results):
    categories = sorted({c for row in results for c in row["by_category"]})
    rows = {}
    for category in categories:
        values = [r["by_category"][category] for r in results if category in r["by_category"]]
        sums = {
            key: sum(r[key] for r in values)
            for key in ("intersection_chars", "predicted_chars", "gold_chars")
        }
        rows[category] = {
            **sums,
            **ratios(sums["intersection_chars"], sums["predicted_chars"], sums["gold_chars"]),
            "events": {k: sum(r["events"][k] for r in values) for k in ("tp", "fp", "fn")},
        }
    sums = {
        key: sum(r[key] for r in rows.values())
        for key in ("intersection_chars", "predicted_chars", "gold_chars")
    }
    return {
        "schema_version": "sc35-span-corpus-score-v1",
        "metric_version": METRIC_VERSION,
        "status": "scored",
        "case_count": len(results),
        "by_category": rows,
        "micro": {
            **sums,
            **ratios(sums["intersection_chars"], sums["predicted_chars"], sums["gold_chars"]),
        },
        "macro_f1": sum(r["f1"] for r in rows.values()) / len(rows) if rows else 1.0,
        "score_percent": 100 * (sum(r["f1"] for r in rows.values()) / len(rows) if rows else 1.0),
        "aggregation": (
            "Macro F1 across categories active in gold or eligible predictions; "
            "character micro reported separately."
        ),
        "empty_policy": (
            "No gold and no predictions: F1/IoU=1; undefined precision/recall remain null."
        ),
        "normative_correctness_verified": False,
    }
