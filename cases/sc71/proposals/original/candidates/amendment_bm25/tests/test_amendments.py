"""Explicit legal references and evidence bundles, using only fictional acts."""

import importlib
import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = (
    "candidate",
    "catalog",
    "relations",
    "lexical",
    "baseline",
    "common",
    "structure",
    "runtime",
)


@pytest.fixture
def api(monkeypatch):
    previous = {name: sys.modules.pop(name) for name in MODULES if name in sys.modules}
    monkeypatch.syspath_prepend(str(PACKAGE))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module = importlib.import_module("candidate")
    yield module
    for name in MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(previous)


def make_corpus(tmp_path, extra=None):
    texts = {
        "a" * 64: (
            "# ПРИКАЗ\n01.01.2024 № 10-1\nО согласовании закупок\n1. Комиссия рассматр"
            "ивает заявку.\n2. Согласование выполняет директор.\n"
        ),
        "b" * 64: (
            "# ПРИКАЗ\n02.02.2025 № 20-1\nО внесении изменения\n1. Внести изменение в "
            "приказ от 01.01.2024 № 10-1: пункт 2 изложить в следующей редакции: «С"
            "огласование выполняет ректор».\n"
        ),
        "c" * 64: (
            "# ПРИКАЗ\n03.03.2025 № 30-1\nОб отчёте\n1. Учесть сведения из приказа от "
            "01.01.2024 № 10-1 при подготовке отчёта.\n"
        ),
    }
    texts.update(extra or {})
    kb = tmp_path / "norm"
    kb.mkdir()
    for origin, text in texts.items():
        (kb / (origin + ".md")).write_text(text, encoding="utf-8", newline="")
    return kb, texts


def test_change_and_mention_are_distinct_and_exact(api, tmp_path):
    kb, texts = make_corpus(tmp_path)
    out = tmp_path / "catalog"
    api.prepare_catalog(kb, out)
    rows = [
        json.loads(x) for x in (out / "relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    change = next(x for x in rows if x["source_origin"] == "b" * 64)
    mention = next(x for x in rows if x["source_origin"] == "c" * 64)
    assert change["relation_type"] == "amends"
    assert change["target_origin"] == "a" * 64
    assert change["target_clauses"] == ["2"]
    assert change["legal_status"] == "unknown"
    assert mention["relation_type"] == "mentions"
    for row in rows:
        assert (
            row["evidence_quote"]
            == texts[row["source_origin"]][row["evidence_start"] : row["evidence_end"]]
        )


def test_ambiguity_and_missing_targets_are_not_resolved(api, tmp_path):
    kb, _ = make_corpus(
        tmp_path,
        {
            "d" * 64: "# ПРИКАЗ\n01.01.2024 № 10-1\nДругой издатель с теми же реквизитами.\n",
            "e" * 64: (
                "# ПРИКАЗ\n01.01.2026 № 90-1\n1. Признать утратившим силу приказ от 0"
                "9.09.2020 № 999.\n"
            ),
        },
    )
    api.prepare_catalog(kb, tmp_path / "catalog")
    rows = [
        json.loads(x)
        for x in (tmp_path / "catalog/relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    change = next(x for x in rows if x["source_origin"] == "b" * 64)
    assert change["resolution"] == "ambiguous" and change["target_origin"] is None
    assert len(change["target_candidates"]) == 2
    missing = next(x for x in rows if x["source_origin"] == "e" * 64)
    assert missing["resolution"] == "missing" and missing["relation_type"] == "repeals"


def test_relations_use_catalog_identities(api, tmp_path):
    kb, _ = make_corpus(tmp_path)
    out = tmp_path / "catalog"
    out.mkdir()
    rows = [
        {
            "origin_id": "a" * 64,
            "identity": {"kind": "order", "date": "01.01.2024", "number": "10-1"},
        },
        {"origin_id": "b" * 64, "identity": None},
    ]
    result = api.relations.build_relations(rows, kb, out)
    assert result["relations"] > 0
    assert result["resolutions"]["unique"] > 0


def test_retrieval_reserves_change_bundle_and_ablation(api, tmp_path, monkeypatch):
    kb, texts = make_corpus(tmp_path)
    index = tmp_path / "index"
    api.build(kb, index, api.defaults())
    monkeypatch.setattr(api.base, "post_json", lambda *a: pytest.fail("retrieval is model-free"))
    result = api.retrieve(index, "Кто выполняет согласование закупок по приказу 10-1?")
    assert result["links"]
    assert any(h["origin_id"] == "b" * 64 and "ректор" in h["quote"] for h in result["hits"])
    for h in result["hits"]:
        assert h["quote"] == texts[h["origin_id"]][h["char_start"] : h["char_end"]]
    for link in result["links"]:
        assert any(
            h["origin_id"] == link["source_origin"]
            and h["char_start"] <= link["evidence_start"]
            and link["evidence_end"] <= h["char_end"]
            for h in result["hits"]
        )
    assert result["context_chars_used"] <= 16000
    assert len(result["hits"]) <= 8 and result["model_calls_retrieval"] == 0
    ablation = api.retrieve(index, "Кто выполняет согласование закупок?", no_relations=True)
    plain = api.lexical.retrieve(index, "Кто выполняет согласование закупок?")
    assert ablation["hits"] == plain["hits"] and ablation["links"] == []


@pytest.mark.parametrize(
    "sentence",
    [
        "1. Не отменять приказ от 01.01.2024 № 10-1.",
        "О внесении изменений в приказ от 01.01.2024 № 10-1",
        "1. Учесть приказ от 01.01.2024 № 10-1.",
    ],
)
def test_titles_negations_and_mentions_do_not_become_changes(api, tmp_path, sentence):
    kb, _ = make_corpus(tmp_path, {"f" * 64: "# ПРИКАЗ\n01.01.2026 № 80-1\n" + sentence + "\n"})
    api.prepare_catalog(kb, tmp_path / "catalog")
    rows = [
        json.loads(x)
        for x in (tmp_path / "catalog/relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(x["relation_type"] == "mentions" for x in rows if x["source_origin"] == "f" * 64)


def test_adjacent_mention_does_not_inherit_amendment(api, tmp_path):
    kb, _ = make_corpus(
        tmp_path,
        {
            "f" * 64: (
                "# ПРИКАЗ\n01.01.2026 № 80-1\n1. Внести изменение в приказ от 01.01.2"
                "024 № 10-1. Учесть приказ от 03.03.2025 № 30-1.\n"
            )
        },
    )
    api.prepare_catalog(kb, tmp_path / "catalog")
    rows = [
        json.loads(x)
        for x in (tmp_path / "catalog/relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected = [x for x in rows if x["source_origin"] == "f" * 64]
    assert [x["relation_type"] for x in selected] == ["amends", "mentions"]


def test_tiny_budget_drops_entire_bundle_without_orphan_link(api, tmp_path):
    kb, _ = make_corpus(tmp_path)
    settings = api.defaults()
    settings["lexical"]["context_chars"] = 1000
    api.build(kb, tmp_path / "index", settings)
    result = api.retrieve(tmp_path / "index", "Кто выполняет согласование закупок?")
    assert result["context_chars_used"] <= 1000
    assert len(result["hits"]) <= settings["lexical"]["top_k"]
    for link in result["links"]:
        assert any(
            h["origin_id"] == link["source_origin"]
            and h["char_start"] <= link["evidence_start"]
            and link["evidence_end"] <= h["char_end"]
            for h in result["hits"]
        )


def test_approval_sheet_does_not_inherit_repeal(api, tmp_path):
    text = (
        "# ПРИКАЗ\n01.01.2026 № 80-1\n"
        "3. Признать утратившим силу прежний состав комиссии.\n"
        "Ректор И.И. Примеров\nЛИСТ СОГЛАСОВАНИЯ\n"
        "к приказу № 10-1 от 01.01.2024\n"
    )
    kb, _ = make_corpus(tmp_path, {"f" * 64: text})
    api.prepare_catalog(kb, tmp_path / "catalog")
    rows = [
        json.loads(x)
        for x in (tmp_path / "catalog/relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(x["relation_type"] == "mentions" for x in rows if x["source_origin"] == "f" * 64)


def test_annex_clause_is_not_misidentified_as_order_clause(api, tmp_path):
    text = (
        "# ПРИКАЗ\n01.01.2026 № 80-1\n"
        "1. Внести изменение в пункт 2 Приложения № 3 к приказу от 01.01.2024 № 10-1.\n"
    )
    kb, _ = make_corpus(tmp_path, {"f" * 64: text})
    api.build(kb, tmp_path / "index", api.defaults())
    result = api.retrieve(tmp_path / "index", "Согласование закупок по приказу 10-1")
    scoped = [x for x in result["links"] if x["source_origin"] == "f" * 64]
    assert scoped and all(not x["target_scope_located"] for x in scoped)
    assert scoped[0]["target_annexes"] == ["3"]
