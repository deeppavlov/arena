"""Arena command metric, one example; no model, filesystem dataset or mean of F1s."""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "runtime"))
from jsonschema import Draft202012Validator  # noqa: E402
from metric import evaluate  # noqa: E402
from taxonomy import validate_taxonomy  # noqa: E402


def score(payload):
    example = payload["example"]
    reference = json.loads(example["reference"])
    if reference["case_id"] != example["id"]:
        raise ValueError("Example/reference identity differs")
    taxonomy = validate_taxonomy(json.loads((BASE / "taxonomy.json").read_text(encoding="utf-8")))
    review = json.loads(payload["answer"])
    schema = json.loads((BASE / "review.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(review)
    result = evaluate(reference, review, taxonomy)
    return {"statistics": {"span_result": result}}


if __name__ == "__main__":
    try:
        json.dump(score(json.load(sys.stdin)), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    except Exception as error:
        sys.stderr.write(f"SC35 metric failed ({type(error).__name__})\n")
        raise SystemExit(1) from None
