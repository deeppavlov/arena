"""Behavioral checks on artificial documents; no real model or dataset."""

import importlib
import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = ("candidate", "catalog", "lexical", "baseline", "common", "structure", "runtime")


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


@pytest.fixture
def corpus(tmp_path):
    kb = tmp_path / "norm"
    kb.mkdir()
    texts = {
        "a" * 64: (
            "# ПРИКАЗ\r\n01.01.2024 № 10-1\r\nО закупках\r\n1. Заявку рассматривает комис"
            "сия.\r\n2. Согласование выполняет директор.\r\n"
        ),
        "b" * 64: (
            "# ПРИКАЗ\n02.02.2025 № 20-1\nОб изменении порядка\n1. Внести изменение в "
            "приказ от 01.01.2024 № 10-1: пункт 2 изложить в следующей редакции: «С"
            "огласование выполняет ректор».\n"
        ),
        "c" * 64: "# ПОЛОЖЕНИЕ\n1. Отпуск оформляется по заявлению.\n",
    }
    for origin, text in texts.items():
        (kb / (origin + ".md")).write_bytes(text.encode())
    manifest = {
        "corpus_version": "synthetic",
        "snapshot_tag": "fixture",
        "objects": [
            {"origin_id": origin, "relative_posix": "отдел/" + origin + ".pdf"} for origin in texts
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return kb, texts


def test_catalog_coordinates_and_primary_identity(api, corpus, tmp_path):
    kb, texts = corpus
    summary = api.prepare_catalog(kb, tmp_path / "catalog")
    assert summary["documents"] == 3
    records = [
        json.loads(x)
        for x in (tmp_path / "catalog/catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record = next(x for x in records if x["origin_id"] == "b" * 64)
    assert record["identity"]["number"] == "20-1"  # not the referenced old order
    assert record["identity"]["date"] == "02.02.2025"
    assert record["original_path"].startswith("отдел/")
    assert record["legal_status"] == "unknown"
    for field, span in record["identity_spans"].items():
        assert texts[record["origin_id"]][span[0] : span[1]] == record["identity"][field]


def test_actual_observations_drive_next_model_step(api, corpus, tmp_path, monkeypatch):
    kb, texts = corpus
    index = tmp_path / "index"
    api.build(kb, index, api.defaults())
    calls = []
    actions = [
        {"tool": "read", "args": {"origin_id": "b" * 64, "line_start": 4, "line_end": 4}},
        {"tool": "lookup", "args": {"number": "10-1", "date": "01.01.2024"}},
        {"tool": "read", "args": {"origin_id": "a" * 64, "line_start": 5, "line_end": 5}},
        {"tool": "finish", "args": {}},
    ]

    def model(url, payload, key, timeout):
        calls.append(json.loads(json.dumps(payload)))
        action = actions[len(calls) - 1]
        return {
            "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    monkeypatch.setattr(api.base, "post_json", model)
    result = api.retrieve(index, "Кто согласует закупку?")
    assert len(calls) == result["model_calls_retrieval"] == 4
    assert "ректор" in calls[1]["messages"][-1]["content"]
    assert result["agent"]["stop_reason"] == "finish"
    assert (
        result["agent"]["observation_chars"] <= api.defaults()["agent"]["total_observation_chars"]
    )
    assert any(h["origin_id"] == "b" * 64 and "ректор" in h["quote"] for h in result["hits"])
    for hit in result["hits"]:
        assert hit["quote"] == texts[hit["origin_id"]][hit["char_start"] : hit["char_end"]]
    assert result["context_chars_used"] <= 16000


def test_tools_reject_paths_and_out_of_bounds_reads(api, corpus, tmp_path):
    kb, _ = corpus
    api.build(kb, tmp_path / "index", api.defaults())
    session = api.ToolSession(tmp_path / "index", api.defaults())
    try:
        bad = session.execute(
            {"tool": "read", "args": {"origin_id": "../secret", "line_start": 1, "line_end": 2}}
        )
        assert bad["status"] == "tool_error"
        bad = session.execute(
            {"tool": "read", "args": {"origin_id": "a" * 64, "line_start": 0, "line_end": 999}}
        )
        assert bad["status"] == "tool_error"
        assert (
            session.execute({"tool": "shell", "args": {"command": "ls"}})["status"] == "tool_error"
        )
        assert session.evidence == []
    finally:
        session.close()


def test_no_agent_is_exact_structural_ablation(api, corpus, tmp_path, monkeypatch):
    kb, _ = corpus
    index = tmp_path / "index"
    api.build(kb, index, api.defaults())
    monkeypatch.setattr(api.base, "post_json", lambda *a: pytest.fail("no model expected"))
    actual = api.retrieve(index, "Кто согласует закупку?", no_agent=True)
    expected = api.lexical.retrieve(index, "Кто согласует закупку?")
    assert actual["hits"] == expected["hits"]
    assert actual["model_calls_retrieval"] == 0


def test_invalid_model_reply_retains_accounting(api, corpus, tmp_path, monkeypatch):
    kb, _ = corpus
    index = tmp_path / "index"
    api.build(kb, index, api.defaults())
    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    monkeypatch.setattr(
        api.base,
        "post_json",
        lambda *a: {
            "choices": [{"message": {"content": "not JSON"}}],
            "usage": {"total_tokens": 9},
        },
    )
    with pytest.raises(api.AgentError) as caught:
        api.retrieve(index, "Кто согласует закупку?")
    assert caught.value.diagnostics["model_calls_retrieval"] == 1
    assert caught.value.diagnostics["calls"][0]["usage"]["total_tokens"] == 9


def test_duplicate_actions_and_budget_stop(api, corpus, tmp_path, monkeypatch):
    kb, _ = corpus
    index = tmp_path / "index"
    api.build(kb, index, api.defaults())
    monkeypatch.setenv("SC71_CHAT_URL", "http://127.0.0.1/v1/chat/completions")
    monkeypatch.setenv("SC71_CHAT_MODEL", "synthetic")
    action = {"tool": "outline", "args": {"origin_id": "a" * 64}}
    monkeypatch.setattr(
        api.base,
        "post_json",
        lambda *a: {"choices": [{"message": {"content": json.dumps(action)}}]},
    )
    result = api.retrieve(index, "Кто согласует закупку?")
    assert result["model_calls_retrieval"] == 4
    assert result["agent"]["stop_reason"] == "step_limit"
    assert any(x["status"] == "duplicate_action" for x in result["agent"]["steps"])


def test_literal_search_is_not_regex_and_read_preserves_crlf(api, corpus, tmp_path):
    kb, texts = corpus
    api.build(kb, tmp_path / "index", api.defaults())
    session = api.ToolSession(tmp_path / "index", api.defaults())
    try:
        result = session.execute(
            {"tool": "grep", "args": {"query": "ЗАЯВКУ", "origin_id": "a" * 64}}
        )
        assert result["hits"] and result["hits"][0]["origin_id"] == "a" * 64
        assert session.execute({"tool": "grep", "args": {"query": "[.*]"}})["hits"] == []
        result = session.execute(
            {"tool": "read", "args": {"origin_id": "a" * 64, "line_start": 4, "line_end": 4}}
        )
        hit = result["hits"][0]
        assert hit["quote"].endswith("\r\n")
        assert hit["quote"] == texts["a" * 64][hit["char_start"] : hit["char_end"]]
    finally:
        session.close()


def test_catalog_does_not_promote_approval_reference_to_identity(api, tmp_path):
    from catalog import primary_identity

    text = "# ПОЛОЖЕНИЕ\nУТВЕРЖДЕНО приказом от 01.01.2024 № 100\n1. Содержание.\n"
    assert primary_identity(text) == (None, {})
    assert primary_identity("Приказом от 01.01.2024 № 100 утверждено положение.\n") == (None, {})


def test_page_separator_is_not_a_source_line_boundary(api, tmp_path):
    from catalog import source_lines

    text = "Первая\fстраница\r\nВторая строка\n"
    assert source_lines(text) == ["Первая\fстраница\r\n", "Вторая строка\n"]


def test_changed_instruction_and_empty_corpus_are_rejected(api, corpus, tmp_path, monkeypatch):
    kb, _ = corpus
    package = tmp_path / "package"
    package.mkdir()
    instruction = package / "AGENT.md"
    instruction.write_bytes((PACKAGE / "AGENT.md").read_bytes())
    monkeypatch.setattr(api, "PACKAGE", package)
    api.build(kb, tmp_path / "index", api.defaults())
    instruction.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="instruction changed"):
        api.retrieve(tmp_path / "index", "Кто согласует закупку?")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="corpus"):
        api.prepare_catalog(empty, tmp_path / "out")
