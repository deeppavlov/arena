"""Stage 3: profile-bounded semantic review, with canonical evidence validation."""

from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "span"))
from projection import VERSION  # noqa: E402

MAX_GATEWAY_RESPONSE = 1024 * 1024
STATUS = {"pass", "violation", "not_applicable", "insufficient_context"}
CONTENT_FALLBACK = {
    "title_sheet_missing",
    "title_designation_missing",
    "approval_designation_suffix",
    "title_year_suffix",
    "title_program_title_case",
    "contents_entry_address_missing",
    "contents_entry_number_missing",
    "contents_entry_title_missing",
    "subsection_case",
    "figure_number_missing",
    "figure_reference",
    "formula_part_prefix",
    "appendix_reference_missing",
    "appendix_toc_missing",
    "empty_section",
    "means_procedure_empty",
}


def semantic_rules(profile):
    rules = [
        r
        for r in profile["rules"]
        if r.get("evidence_channel") == "semantic"
        or r["rule_id"].split("-", 4)[-1] in CONTENT_FALLBACK
    ]
    if not rules:
        raise ValueError("The profile has no supported content requirements")
    return rules


def request_for(document, taxonomy, profile):
    if (
        document.get("extraction_version") != VERSION
        or document.get("offset_unit") != "unicode_codepoint"
        or document.get("taxonomy_sha256") != taxonomy["taxonomy_sha256"]
    ):
        raise ValueError("Semantic input uses incompatible coordinates/taxonomy")
    rules = semantic_rules(profile)
    requirements = [
        {
            k: r.get(k, r["requirement"] if k == "title" else None)
            for k in ("rule_id", "standard", "clause", "title", "requirement", "coverage_level")
        }
        for r in rules
    ]
    skill = (BASE / "skills/espd-content-review/SKILL.md").read_text(encoding="utf-8")
    system = (
        skill
        + "\n\nReturn exactly this JSON shape, no Markdown:\n"
        + json.dumps(
            {
                "assessments": [
                    {
                        "rule_id": "one supplied ID",
                        "status": "pass|violation|not_applicable|insufficient_context",
                        "reason": "reason",
                    }
                ],
                "findings": [
                    {
                        "rule_id": "violated supplied ID",
                        "message": "specific observation",
                        "spans": [{"start": 0, "end": 1}],
                        "evidence_quotes": ["exact selected text"],
                    }
                ],
            },
            ensure_ascii=False,
        )
    )
    payload = {
        "case_id": document["case_id"],
        "requirements": requirements,
        "document": {
            k: document[k] for k in ("text", "paragraphs", "offset_unit", "extraction_version")
        },
    }
    if "formal_evidence" in document:
        payload["formal_evidence"] = document["formal_evidence"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    if len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) > 15 * 1024 * 1024:
        raise ValueError("Semantic request exceeds gateway transport size; no silent truncation")
    return messages


def normalize(raw, document, taxonomy, profile):
    if not isinstance(raw, dict) or set(raw) != {"assessments", "findings"}:
        raise ValueError("Semantic response must contain assessments and findings only")
    rules = {r["rule_id"]: r for r in semantic_rules(profile)}
    assessments = {}
    if not isinstance(raw["assessments"], list) or not isinstance(raw["findings"], list):
        raise ValueError("Semantic arrays are malformed")
    for item in raw["assessments"]:
        if not isinstance(item, dict) or set(item) != {"rule_id", "status", "reason"}:
            raise ValueError("Invalid semantic assessment")
        key = item["rule_id"]
        if key not in rules or key in assessments or item["status"] not in STATUS:
            raise ValueError("Unknown, duplicate or invalid semantic assessment")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("Semantic assessment needs a reason")
        assessments[key] = item
    if set(assessments) != set(rules):
        raise ValueError("Semantic response did not account for every supplied rule")
    text, findings, violated = document["text"], [], set()
    for item in raw["findings"]:
        if not isinstance(item, dict) or set(item) != {
            "rule_id",
            "message",
            "spans",
            "evidence_quotes",
        }:
            raise ValueError("Invalid semantic finding")
        key = item["rule_id"]
        if key not in rules or assessments[key]["status"] != "violation":
            raise ValueError("Semantic finding contradicts its assessment")
        if not isinstance(item["message"], str) or not item["message"].strip():
            raise ValueError("Semantic finding needs an observation")
        spans, quotes = item["spans"], item["evidence_quotes"]
        if (
            not isinstance(spans, list)
            or not spans
            or not isinstance(quotes, list)
            or len(spans) != len(quotes)
        ):
            raise ValueError("Every semantic span needs an evidence quote")
        for span, quote in zip(spans, quotes, strict=True):
            if not isinstance(span, dict) or set(span) != {"start", "end"}:
                raise ValueError("Malformed semantic span")
            start, end = span["start"], span["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("Semantic span is outside canonical Unicode text")
            if not isinstance(quote, str) or text[start:end] != quote:
                raise ValueError("Semantic evidence quote differs from canonical span")
        rule = rules[key]
        findings.append(
            {
                "id": f"S{len(findings) + 1:04d}",
                "rule_id": key,
                "category": taxonomy["rule_categories"][key],
                "spans": spans,
                "message": item["message"],
                "evidence_quotes": quotes,
                "anchor_kind": "semantic_object",
                "basis_kind": "gost_requirement",
                "normative_basis": [
                    {
                        "standard": rule["standard"],
                        "clause": rule["clause"],
                        "source": rule.get("source"),
                    }
                ],
            }
        )
        violated.add(key)
    if violated != {key for key, a in assessments.items() if a["status"] == "violation"}:
        raise ValueError("A semantic violation has no located finding")
    return {
        "schema_version": "sc35-span-review-v1",
        "case_id": document["case_id"],
        "document_sha256": document["document_sha256"],
        "text_sha256": document["text_sha256"],
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "extraction_version": VERSION,
        "offset_unit": "unicode_codepoint",
        "source_profile": "espd-pmi",
        "adapter": "semantic-llm",
        "status": "ready",
        "unresolved": [],
        "findings": findings,
        "semantic_assessments": list(assessments.values()),
        "limits": [
            "Only supplied semantic profile predicates; "
            "no external specification or rendered geometry.",
            "Insufficient-context and not-applicable decisions are abstentions, "
            "not compliance certificates.",
        ],
    }


def review(document, taxonomy, profile, client):
    messages = request_for(document, taxonomy, profile)
    started = time.monotonic()
    response = client.complete(messages)
    # Parse failures are candidate errors, not a reason to generate another valid answer.
    if not isinstance(response, str):
        raise ValueError("Semantic completion must be JSON text")
    result = normalize(json.loads(response), document, taxonomy, profile)
    result["semantic_transport"] = {
        "llm_called": bool(getattr(client, "is_live", False)),
        "calls": 1,
        "seconds": time.monotonic() - started,
    }
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Gateway redirects are disabled")


class GatewayClient:
    """One request to the arena-provided internal gateway; it owns retries/usage."""

    is_live = True

    def __init__(self, payload, opener=None):
        self.model = payload.get("model")
        self.timeout = payload.get("model_request_timeout")
        endpoint = payload.get("model_base_url")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or not isinstance(endpoint, str)
        ):
            raise ValueError("Arena model configuration is required")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"/runs/[^/]+/[^/]+/v1/?", parsed.path)
        ):
            raise ValueError("Expected a scoped internal arena model gateway")
        if (
            type(self.timeout) not in {int, float}
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("Arena model_request_timeout must be positive")
        self._endpoint = endpoint.rstrip("/") + "/chat/completions"
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def complete(self, messages):
        body = json.dumps({"model": self.model, "messages": messages}, ensure_ascii=False).encode(
            "utf-8"
        )
        request = urllib.request.Request(
            self._endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = response.read(MAX_GATEWAY_RESPONSE + 1)
        except (OSError, urllib.error.URLError):
            raise RuntimeError(
                "Internal model gateway request failed; scoped URL omitted"
            ) from None
        if len(data) > MAX_GATEWAY_RESPONSE:
            raise ValueError("Gateway response exceeds transport limit")
        try:
            result = json.loads(data)
            choices = result["choices"]
            if len(choices) != 1 or choices[0].get("finish_reason", "stop") != "stop":
                raise ValueError("Expected one complete final model response")
            content = choices[0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty model response")
            return content
        except (ValueError, TypeError, KeyError, IndexError):
            raise ValueError("Invalid final response from model gateway") from None
