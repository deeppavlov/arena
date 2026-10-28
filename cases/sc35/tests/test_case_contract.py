"""Portable synthetic checks; no original documents, external model or arena services."""

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET
from zipfile import ZipFile

CASE = Path(__file__).resolve().parents[1]
WAY = CASE / "ways/baseline"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def synthetic_input(empty_object=False):
    names = [
        "Объект испытаний",
        "Цель испытаний",
        "Требования\u00a0к\u00a0программе",
        "Требования\u00a0к\u00a0программной документации",
        "Средства\u00a0и\u00a0порядок испытаний",
        "Методы испытаний",
    ]
    texts = []
    for index, name in enumerate(names):
        texts.append(name)
        if index != 0 or not empty_object:
            texts.append("Описание.")
    root = ET.Element(W + "document")
    body = ET.SubElement(root, W + "body")
    for text in texts:
        ET.SubElement(ET.SubElement(ET.SubElement(body, W + "p"), W + "r"), W + "t").text = text
    buffer = BytesIO()
    with ZipFile(buffer, "w") as package:
        package.writestr("word/document.xml", ET.tostring(root))
    item = {
        "case_id": "synthetic",
        "docx_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "document_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "text_sha256": hashlib.sha256(("\n".join(texts) + "\n").encode()).hexdigest(),
    }
    return {
        "input": json.dumps(item, ensure_ascii=False),
        "model": "internal-fixture-model",
        "model_base_url": "http://model-gateway/runs/synthetic/fixture-capability/v1",
        "model_request_timeout": 10,
    }


class FakeClient:
    is_live = False
    calls = 0

    def __init__(self, payload):
        self.payload = payload

    def complete(self, messages):
        type(self).calls += 1
        data = json.loads(messages[1]["content"])
        text, requirements = data["document"]["text"], data["requirements"]
        key = requirements[0]["rule_id"]
        quote = "Объект испытаний"
        start = text.index(quote)
        return json.dumps(
            {
                "assessments": [
                    {
                        "rule_id": r["rule_id"],
                        "status": "violation" if r["rule_id"] == key else "pass",
                        "reason": "Synthetic fixture; not a real model assessment.",
                    }
                    for r in requirements
                ],
                "findings": [
                    {
                        "rule_id": key,
                        "message": "Synthetic semantic observation",
                        "spans": [{"start": start, "end": start + len(quote)}],
                        "evidence_quotes": [quote],
                    }
                ],
            }
        )


class ContractTests(unittest.TestCase):
    def test_candidate_catalog_excludes_document_reference_inventory(self):
        path = WAY / "runtime/sc35/standards-graph/catalog.json"
        catalog = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse("document_reference_inventory" in catalog)

    def test_runtime_dataset_helper_contains_no_reference_generator(self):
        helper = module("sc35_runtime_save", WAY / "runtime/sc35/span/build_dataset.py")
        self.assertFalse(hasattr(helper, "build"))
        self.assertFalse(hasattr(helper, "reference_finding"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/result.json"
            helper.save(path, {"synthetic": "пример"})
            self.assertEqual(json.loads(path.read_text()), {"synthetic": "пример"})

    def test_arena_command_processes_preserve_corpus_f1_and_no_model_usage(self):
        from evaluator.contracts import Example
        from evaluator.metric_package import freeze_metric, load_metric
        from evaluator.metrics import aggregate_results, evaluate_answer

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "metric"
            freeze_metric(CASE / "metric", package)
            metric = load_metric(package)
            self.assertFalse(metric.command.uses_model)
            self.assertFalse(metric.aggregation.uses_model)
            results, usage = [], {}
            for key, gold, predicted in (("a", 10, 5), ("b", 1, 1)):
                fixture = self.example(key, gold, predicted)
                result = evaluate_answer(
                    metric, Example(**fixture["example"]), fixture["answer"], usage=usage,
                )
                results.append({"example_id": key, **result})
            self.assertAlmostEqual(aggregate_results(metric, results, usage=usage), 100 * 12 / 17)
            self.assertEqual(usage, {})

    def test_solver_transport_limit_includes_trailing_newline(self):
        candidate = module("sc35_portable_transport", WAY / "solve.py")
        payload = synthetic_input()
        item = json.loads(payload["input"])
        review = {"x": ""}
        pipeline = SimpleNamespace(
            load_assets=lambda _: ({}, {}),
            Projection=lambda _: SimpleNamespace(
                source_sha256=item["document_sha256"], text_sha256=item["text_sha256"],
            ),
            review_document=lambda *_: review,
        )
        empty_answer = json.dumps(review, ensure_ascii=False, separators=(",", ":"))
        overhead = len(json.dumps({"answer": empty_answer}, ensure_ascii=False).encode())
        with patch.object(candidate, "load_pipeline", return_value=pipeline):
            review["x"] = "a" * (1024 * 1024 - overhead - 1)
            result = candidate.solve(payload, WAY, FakeClient)
            self.assertEqual(len((json.dumps(result) + "\n").encode()), 1024 * 1024)
            review["x"] += "a"
            with self.assertRaises(ValueError):
                candidate.solve(payload, WAY, FakeClient)

    def test_model_pass_does_not_erase_a_confirmed_instrumental_violation(self):
        candidate = module("sc35_portable_conflict", WAY / "solve.py")
        response = candidate.solve(synthetic_input(empty_object=True), WAY, FakeClient)
        review = json.loads(response["answer"])
        rule = "ESPD-19.301-79-2.1-empty_section"
        assessed = next(r for r in review["rule_assessments"] if r["rule_id"] == rule)
        self.assertEqual(assessed["status"], "violations_found")
        self.assertEqual(assessed["model_assessment"]["status"], "pass")
        self.assertTrue(any(f["rule_id"] == rule for f in review["findings"]))

    def test_three_stages_and_one_injected_model_response(self):
        FakeClient.calls = 0
        candidate = module("sc35_portable_solver", WAY / "solve.py")
        response = candidate.solve(synthetic_input(), WAY, FakeClient)
        review = json.loads(response["answer"])
        self.assertEqual(review["stages"], ["graph", "format", "typography", "semantic-llm"])
        self.assertEqual(FakeClient.calls, 1)
        self.assertEqual(len(review["semantic_assessments"]), 40)
        self.assertFalse(review["llm_called"])
        self.assertTrue(review["semantic_stage_enabled"])

    def test_input_identity_mismatch_is_not_a_clean_document(self):
        candidate = module("sc35_portable_bad_input", WAY / "solve.py")
        payload = synthetic_input()
        data = json.loads(payload["input"])
        data["text_sha256"] = "0" * 64
        payload["input"] = json.dumps(data)
        with self.assertRaises(ValueError):
            candidate.solve(payload, WAY, FakeClient)

    def test_isolated_process_from_root_has_no_deh_import_dependency(self):
        script = (
            "import json,runpy,sys; n=runpy.run_path(sys.argv[1],run_name='fixture'); "
            "s=n['module']('fresh_solver',n['WAY']/'solve.py'); "
            "print(json.dumps(s.solve(json.load(sys.stdin),n['WAY'],n['FakeClient'])))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(Path(__file__).resolve())],
            input=json.dumps(synthetic_input()),
            text=True,
            capture_output=True,
            cwd=Path(Path.cwd().anchor),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("semantic-llm", json.loads(json.loads(result.stdout)["answer"])["stages"])

    def example(self, case_id, gold_end, predicted_end):
        taxonomy = json.loads((CASE / "metric/taxonomy.json").read_text(encoding="utf-8"))
        rule = next(
            key
            for key, category in taxonomy["rule_categories"].items()
            if category == "sections_content"
        )
        identity = {
            "case_id": case_id,
            "document_sha256": "a" * 64,
            "text_sha256": "b" * 64,
            "taxonomy_sha256": taxonomy["taxonomy_sha256"],
            "extraction_version": "sc35-corrupted-text-v1",
            "offset_unit": "unicode_codepoint",
        }
        finding = {
            "id": "F1",
            "rule_id": rule,
            "category": "sections_content",
            "message": "Synthetic",
            "spans": [{"start": 0, "end": gold_end}],
            "anchor_kind": "object",
        }
        reference = {
            **identity,
            "schema_version": "sc35-span-reference-v1",
            "text_length": 100,
            "findings": [finding],
            "evaluation_scope": {"rule_ids": [rule], "regions": [{"start": 0, "end": 100}]},
        }
        answer = {
            **identity,
            "schema_version": "sc35-span-review-v1",
            "status": "ready",
            "unresolved": [],
            "source_profile": "espd-pmi",
            "findings": [{**finding, "spans": [{"start": 0, "end": predicted_end}]}],
        }
        return {
            "example": {"id": case_id, "input": "synthetic", "reference": json.dumps(reference)},
            "answer": json.dumps(answer),
        }

    def test_metric_corpus_f1_is_not_mean_of_per_example_f1(self):
        score = module("sc35_portable_metric", CASE / "metric/score.py")
        aggregate = module("sc35_portable_aggregate", CASE / "metric/aggregate.py")
        results = [
            {"example_id": key, **score.score(self.example(key, gold, predicted))}
            for key, gold, predicted in (("a", 10, 5), ("b", 1, 1))
        ]
        self.assertAlmostEqual(aggregate.combine({"results": results})["score"], 100 * 12 / 17)

    def test_invalid_span_is_metric_failure_not_zero_score(self):
        score = module("sc35_portable_invalid_metric", CASE / "metric/score.py")
        with self.assertRaises(ValueError):
            score.score(self.example("a", 10, 101))

    def test_duplicate_or_empty_aggregation_is_rejected(self):
        score = module("sc35_portable_duplicate_score", CASE / "metric/score.py")
        aggregate = module("sc35_portable_duplicate_aggregate", CASE / "metric/aggregate.py")
        row = {"example_id": "a", **score.score(self.example("a", 10, 5))}
        for values in ([], [row, row]):
            with self.assertRaises(ValueError):
                aggregate.combine({"results": values})


if __name__ == "__main__":
    unittest.main()
