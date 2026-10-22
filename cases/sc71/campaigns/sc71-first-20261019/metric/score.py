import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

TOPICS = {"T1", "T2", "T3", "T4", "T5"}
ORIGIN = re.compile(r"^[0-9a-f]{64}$")
LINK = re.compile(r"/kb/norm/([0-9a-f]{64})\.md\b", re.IGNORECASE)
QUOTES = re.compile(r'«(.*?)»|“(.*?)”|„(.*?)“|"([^"]*)"', re.DOTALL)
UNICODE_QUOTES = re.compile(r"«(.*?)»|“(.*?)”|„(.*?)“", re.DOTALL)
REFUSAL = "ответ отсутствует в базе"
SCORE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["score", "reason"],
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string", "minLength": 1, "pattern": r"\S"},
    },
}


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def identifiers(passage: dict[str, Any]) -> set[str]:
    values = [passage["origin_id"], *passage.get("equivalent_origin_ids", [])]
    if any(not isinstance(value, str) or not ORIGIN.fullmatch(value) for value in values):
        raise ValueError("invalid reference document identifier")
    return set(values)


def reference_data(payload: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(payload["example"]["reference"])
    if (
        not isinstance(reference, dict)
        or reference.get("format") != "sc71-reference-v1"
        or reference.get("topic") not in TOPICS
        or not isinstance(reference.get("reference_markdown"), str)
        or not reference["reference_markdown"].strip()
        or not isinstance(reference.get("gold"), dict)
    ):
        raise ValueError("invalid SC-71 reference wrapper")
    gold = reference["gold"]
    if gold.get("topic") != reference["topic"] or type(gold.get("answerable")) is not bool:
        raise ValueError("invalid SC-71 reference topic or answerability")
    for key in ("citations", "must_include", "must_not_include"):
        if not isinstance(gold.get(key), list):
            raise ValueError("invalid SC-71 reference fields")
    for key in ("must_include", "must_not_include"):
        if any(not isinstance(value, str) or not value.strip() for value in gold[key]):
            raise ValueError("invalid SC-71 reference requirement")
    return reference


def json_answer(answer: str) -> dict[str, Any] | None:
    text = answer.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def answer_quotes(answer: str) -> list[str]:
    structured = json_answer(answer)
    found: list[str] = []

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                collect(child, name)
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, str):
            if key.casefold() in {"quote", "цитата", "citation_quote"}:
                found.append(value)
            else:
                found.extend(next(part for part in match if part) for match in
                             UNICODE_QUOTES.findall(value) if any(match))

    if structured is not None:
        collect(structured)
    else:
        found.extend(next(part for part in match if part) for match in
                     QUOTES.findall(answer) if any(match))
    return list(dict.fromkeys(value for value in found if len(normalized(value)) >= 20))


def passages(gold: dict[str, Any]) -> list[dict[str, Any]]:
    items = list(gold["citations"])
    if gold["topic"] == "T1" and "t1" in gold:
        items.append(gold["t1"]["supersession_evidence"])
    if gold["topic"] == "T3":
        items.extend(gold["t3"]["steps"])
    if gold["topic"] == "T4":
        items.extend(gold["t4"][key] for key in ("norm_a", "norm_b"))
    return items


def prepare_corpus(
    gold: dict[str, Any], kb_root: Path, cited: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]], set[str]]:
    corpus = kb_root / "norm"
    if not corpus.is_dir() or not any(corpus.glob("*.md")):
        raise ValueError("SC-71 corpus is missing or empty")
    documents: dict[str, str] = {}
    evidence: list[dict[str, Any]] = []
    missing = set()

    def read(origin: str, required: bool) -> str | None:
        if origin in documents:
            return documents[origin]
        path = corpus / f"{origin}.md"
        if not path.is_file():
            if required:
                raise ValueError("reference document missing from frozen corpus")
            missing.add(origin)
            return None
        if not path.resolve().is_relative_to(corpus.resolve()):
            raise ValueError("corpus document escapes frozen corpus")
        documents[origin] = path.read_text(encoding="utf-8")
        return documents[origin]

    seen = set()
    for passage in passages(gold):
        allowed = identifiers(passage)
        for origin in sorted(allowed):
            read(origin, True)
        quote = passage["quote"]
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("invalid reference quote")
        origin = passage["origin_id"]
        match = re.search(r"\s+".join(re.escape(word) for word in quote.split()),
                          documents[origin], re.IGNORECASE)
        if match is None:
            raise ValueError("reference quote missing from frozen corpus")
        key = (origin, match.group())
        if key not in seen:
            evidence.append({"origin_id": origin, "equivalent_origin_ids": sorted(allowed),
                             "locator": passage["locator"], "quote": match.group()})
            seen.add(key)
    for origin in sorted(cited):
        read(origin, False)
    return documents, evidence, missing


def diagnostics(
    answer: str, gold: dict[str, Any], cited: set[str], documents: dict[str, str],
    missing: set[str],
) -> dict[str, Any]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    abstained = bool(lines) and normalized(lines[0]).rstrip(".").strip() == REFUSAL
    answerable = gold["answerable"]
    quotes = answer_quotes(answer)
    cited_texts = [normalized(documents[origin]) for origin in cited if origin in documents]
    fabricated = any(not any(normalized(quote) in text for text in cited_texts) for quote in quotes)
    requirements = gold["t3"]["steps"] if gold["topic"] == "T3" else [
        item for item in gold["citations"] if item["role"] == "primary"
    ]
    required_ids = {frozenset(identifiers(item)) for item in requirements}
    permitted: set[str] = set()
    for item in gold["citations"]:
        if item["role"] in {"primary", "supporting", "superseded"}:
            permitted.update(identifiers(item))
    if answerable and not required_ids:
        raise ValueError("answerable reference has no required documents")
    recall = mean(bool(group & cited) for group in required_ids) if required_ids else None
    included = gold["must_include"]
    answer_normalized = normalized(answer)
    include_hit = (
        mean(normalized(value) in answer_normalized for value in included) if included else 1
    )
    forbidden = any(normalized(value) in answer_normalized for value in gold["must_not_include"])
    result: dict[str, Any] = {
        "topic": gold["topic"], "answerable": answerable,
        "candidate_failure": not bool(answer.strip()),
        "cited_document_count": len(cited), "nonexistent_citation": bool(missing),
        "citation_recall": recall if answerable else None,
        "citation_precision": len(cited & permitted) / len(cited) if answerable and cited else
                              (0.0 if answerable else None),
        "fabricated_quote": fabricated, "quote_count": len(quotes),
        "must_include_hit": include_hit, "must_not_include_hit": forbidden,
        "abstained": abstained, "false_abstention": answerable and abstained,
        "hallucination": (not answerable and not abstained) or bool(missing),
        "grounded": recall == 1 and not fabricated and not missing if answerable else None,
    }
    if gold["topic"] == "T1":
        result["t1_update_error"] = forbidden and include_hit < 1
    if gold["topic"] == "T4":
        structured = json_answer(answer) or {}
        target = gold["t4"]
        result["collision_type_match"] = (
            structured.get("collision_type") == target["collision_type"]
        )
        applies = ["a", "b"] if target["applicable"] == "both" else [target["applicable"]]
        applicable_ids = set().union(*(identifiers(target[f"norm_{key}"]) for key in applies))
        value = structured.get("applicable_norm")
        serialized = (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        )
        selected = set(re.findall(r"\b[0-9a-f]{64}\b", serialized, re.IGNORECASE)) if isinstance(
            serialized, str,
        ) else set()
        result["applicable_norm_match"] = bool(
            {value.lower() for value in selected} & applicable_ids,
        )
    if gold["topic"] == "T5":
        target = gold["t5"]["target_origin_id"]
        allowed = {target}
        for item in gold["citations"]:
            if target in identifiers(item):
                allowed.update(identifiers(item))
        result["target_hit"] = bool(allowed & cited)
    return result


def complete_http(payload: dict[str, Any], body: dict[str, Any]) -> str:
    request = urllib.request.Request(
        payload["model_base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=payload["model_request_timeout"]) as response:
        result = json.load(response)
    choice = result["choices"][0]
    if choice.get("finish_reason", "stop") != "stop" or not isinstance(
        choice["message"]["content"], str,
    ):
        raise ValueError("judge response did not complete")
    return choice["message"]["content"]


def evaluate(
    payload: dict[str, Any], kb_root: Path,
    complete: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    policy = json.loads(Path(__file__).with_name("policy.json").read_text(encoding="utf-8"))
    if payload["model"] != policy["model"]:
        raise ValueError("SC-71 judge model differs from frozen model")
    reference = reference_data(payload)
    answer = payload["answer"]
    if not isinstance(answer, str):
        raise ValueError("answer must be text")
    cited = {origin.lower() for origin in LINK.findall(answer)}
    documents, evidence, missing = prepare_corpus(reference["gold"], kb_root, cited)
    stats = diagnostics(answer, reference["gold"], cited, documents, missing)
    if stats["candidate_failure"] or stats["hallucination"]:
        return {"score": 0.0, "statistics": stats}
    scores = {}
    weighted = []
    for criterion in policy["criteria"]:
        votes = []
        user_data = {
            "criterion": {"id": criterion["id"], "question": criterion["question"]},
            "question": payload["example"]["input"],
            "reference": reference["reference_markdown"],
            "verified_source_quotes": evidence, "answer": answer, "checks": stats,
        }
        body = {
            "model": policy["model"], "reasoning_effort": policy["reasoning_effort"],
            "response_schema": SCORE_SCHEMA,
            "messages": [
                {"role": "system", "content": policy["instruction"]},
                {"role": "user", "content": json.dumps(user_data, ensure_ascii=False)},
            ],
        }
        for index in range(policy["votes"]):
            request = {
                **body,
                "trace_metadata": {
                    "role": "judge", "criterion": criterion["id"], "vote": index + 1,
                },
            }
            response = (
                complete(request) if complete is not None else complete_http(payload, request)
            )
            value = json.loads(response)
            if (
                not isinstance(value, dict) or set(value) != {"score", "reason"}
                or type(value["score"]) is not int or not 1 <= value["score"] <= 5
                or not isinstance(value["reason"], str) or not value["reason"].strip()
            ):
                raise ValueError("invalid judge vote")
            votes.append(value["score"])
        scores[criterion["id"]] = votes
        weighted.append(criterion["weight"] * mean(votes))
    stats["judge_scores"] = scores
    return {"score": math.fsum(weighted), "statistics": stats}


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    topics: dict[str, list[float]] = defaultdict(list)
    for result in results:
        topic = result["statistics"]["topic"]
        score = result["score"]
        if topic not in TOPICS or type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("invalid topic measurement")
        if not 0 <= score <= 5:
            raise ValueError("SC-71 score must be between zero and five")
        topics[topic].append(score)
    if not topics:
        raise ValueError("cannot aggregate empty measurements")
    return {"score": mean(mean(scores) for scores in topics.values())}


def main() -> None:
    payload = json.load(sys.stdin)
    if sys.argv[1:] == ["--aggregate"]:
        result = aggregate(payload["results"])
    elif not sys.argv[1:]:
        result = evaluate(payload, Path(os.environ["ARENA_SC71_KB_ROOT"]))
    else:
        raise ValueError("unsupported metric arguments")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
