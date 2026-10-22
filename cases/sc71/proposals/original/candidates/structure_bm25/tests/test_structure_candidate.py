"""Source-coordinate and retrieval contracts on invented documents only."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def api():
    sys.path.insert(0, str(PACKAGE))
    spec = importlib.util.spec_from_file_location(
        "new_structure_candidate", PACKAGE / "candidate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus(folder, docs):
    folder.mkdir()
    for origin, text in docs.items():
        (folder / f"{origin}.md").write_bytes(text.encode("utf-8"))
    return folder


def test_tree_preserves_crlf_subclauses_and_ignores_page_numbers(api):
    text = (
        "# ПОЛОЖЕНИЕ\r\n1. Общие положения\r\n"
        "1.1. Заявление подаётся комиссии.\r\nПродолжение условия.\r\n"
        "12\r\n1.2. Исключения рассматривает директор.\r\n"
    )
    parsed = api.parse_document(text, 1800)
    nodes = parsed["nodes"]
    clauses = {n["label"]: n for n in nodes if n["kind"] == "clause"}
    assert set(clauses) == {"1", "1.1", "1.2"}
    assert clauses["1.1"]["parent"] == clauses["1"]["id"]
    assert parsed["metadata"]["doc_type"] == "regulation"
    assert all(c["text"] == text[c["start"] : c["end"]] for c in parsed["chunks"])
    assert any("Продолжение условия" in c["text"] for c in parsed["chunks"])
    assert all(c["line_start"] == text.count("\n", 0, c["start"]) + 1 for c in parsed["chunks"])


def test_generic_prose_is_paragraph_based_and_long_lines_are_covered(api):
    text = "Обычный текст\n\n" + "длинно " * 300 + "\n\n2026 год\nКонец."
    parsed = api.parse_document(text, 200)
    assert parsed["metadata"]["doc_type"] == "generic"
    assert all(len(c["text"]) <= 200 for c in parsed["chunks"])
    covered = set()
    for c in parsed["chunks"]:
        covered.update(range(c["start"], c["end"]))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    assert not any(n["label"] == "2026" for n in parsed["nodes"])


@pytest.mark.parametrize("heading,expected", [("# ПРИКАЗ", "order"), ("# ДОГОВОР", "contract")])
def test_document_profiles(api, heading, expected):
    parsed = api.parse_document(heading + "\n1. Обязательное условие.\n", 1800)
    assert parsed["metadata"]["doc_type"] == expected
    assert any(n["kind"] == "clause" for n in parsed["nodes"])


def test_expansion_restores_intro_and_exception_and_keeps_exact_spans(api, tmp_path):
    origin = "a" * 64
    text = (
        "# ПОЛОЖЕНИЕ\n1. Правила оформления\n"
        "1.1. Обычную заявку направляют комиссии.\n"
        "1.2. Срочную заявку направляют директору.\n"
        "2. Архив\n2.1. Копии сохраняются.\n"
    )
    kb = corpus(tmp_path / "norm", {origin: text, "b" * 64: "# Новости\nПогода ясная.\n"})
    config = api.defaults()
    config["structure"]["parent_chars"] = 1200
    api.build(kb, tmp_path / "index", config)
    result = api.retrieve(tmp_path / "index", "Кому направляют срочную заявку?")
    first = result["hits"][0]
    assert "1.1." in first["quote"] and "1.2." in first["quote"]
    assert "2. Архив" not in first["quote"]
    assert first["quote"] == text[first["char_start"] : first["char_end"]]
    assert first["seed_char_start"] >= first["char_start"]
    assert first["seed_char_end"] <= first["char_end"]
    assert first["source_relpath"] == origin + ".md"
    assert result["context_chars_used"] <= config["lexical"]["context_chars"]
    assert (kb / (origin + ".md")).read_bytes().decode("utf-8") == text
    leaf = api.retrieve(tmp_path / "index", "Кому направляют срочную заявку?", expand=False)
    assert leaf["structure_expansion"] is False
    assert leaf["hits"][0]["quote"] != first["quote"]


def test_long_table_retains_original_header_and_row(api, tmp_path):
    origin = "c" * 64
    text = "# ПОЛОЖЕНИЕ\n1. Таблица сроков\n" + "Введение. " * 60 + "\n\n"
    header = "| Событие | Срок |\n| --- | --- |\n"
    text += header + "".join(f"| Событие {i} | {i} дней |\n" for i in range(35))
    config = api.defaults()
    config["lexical"]["chunk_chars"] = 180
    config["lexical"]["max_per_document"] = 8
    config["structure"]["parent_chars"] = 220
    api.build(corpus(tmp_path / "norm", {origin: text}), tmp_path / "index", config)
    result = api.retrieve(tmp_path / "index", "Событие 34 срок")
    assert any("Событие 34" in h["quote"] for h in result["hits"])
    assert any(header.strip() in h["quote"] for h in result["hits"])
    assert all(h["quote"] == text[h["char_start"] : h["char_end"]] for h in result["hits"])


def test_small_context_falls_back_without_truncating_a_quote(api, tmp_path):
    origin = "d" * 64
    text = "# ПОЛОЖЕНИЕ\n1. Условия\n1.1. Заявку принимает комиссия.\n1.2. " + "Сведения. " * 70
    config = api.defaults()
    config["lexical"]["context_chars"] = 500
    api.build(corpus(tmp_path / "norm", {origin: text}), tmp_path / "index", config)
    result = api.retrieve(tmp_path / "index", "Заявку принимает комиссия")
    assert result["hits"]
    assert result["context_chars_used"] <= 500
    assert all(h["quote"] == text[h["char_start"] : h["char_end"]] for h in result["hits"])


def test_empty_query_no_matches_rebuild_and_wrong_package_fail(api, tmp_path):
    kb = corpus(tmp_path / "norm", {"e" * 64: "# ПРИКАЗ\n1. Комиссия рассматривает заявки.\n"})
    output = tmp_path / "index"
    api.build(kb, output, api.defaults())
    assert api.retrieve(output, "ксилофон")["hits"] == []
    with pytest.raises(ValueError):
        api.retrieve(output, " ")
    with pytest.raises(ValueError):
        api.build(kb, output, api.defaults())
    manifest = json.loads((output / "build.json").read_text(encoding="utf-8"))
    manifest["candidate"] = "planned_bm25"
    (output / "build.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate"):
        api.retrieve(output, "заявка")


def test_standalone_cli_reads_index_without_writing_it(api, tmp_path):
    kb = corpus(tmp_path / "norm", {"f" * 64: "# ПРИКАЗ\n1. Комиссия согласует закупку.\n"})
    build = tmp_path / "index"
    api.build(kb, build, api.defaults())
    before = {p.name: p.read_bytes() for p in build.iterdir()}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE / "run.py"),
            "--build-dir",
            str(build),
            "--question",
            "Кто согласует закупку?",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=True,
    )
    assert json.loads(run.stdout)["method"] == "structure_bm25"
    trace = json.loads(run.stderr)
    assert "quote" not in json.dumps(trace)
    assert {p.name: p.read_bytes() for p in build.iterdir()} == before


def test_configuration_tampering_and_index_corruption_fail(api, tmp_path, monkeypatch):
    kb = corpus(tmp_path / "norm", {"a" * 64: "# ПРИКАЗ\n1. Комиссия согласует заявку.\n"})
    output = tmp_path / "index"
    api.build(kb, output, api.defaults())
    manifest = json.loads((output / "build.json").read_text(encoding="utf-8"))
    manifest["config"] = dict(manifest["config"], top_k=100)
    (output / "build.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        api.retrieve(output, "Комиссия")
    manifest["config"]["top_k"] = 8
    (output / "build.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (output / "index.sqlite3").open("ab") as stream:
        stream.write(b"synthetic corruption")
    monkeypatch.setenv("SC71_VERIFY_BUILD", "1")
    with pytest.raises(ValueError, match="hash mismatch"):
        api.retrieve(output, "Комиссия")
