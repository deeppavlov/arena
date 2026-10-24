import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval import documents, storage


def test_build_keeps_source_paths_without_checksums(tmp_path):
    kb = tmp_path / "norm"
    kb.mkdir()
    origin = "a" * 64
    (kb / f"{origin}.md").write_text("# Правила\nЗаявку подают комиссии.\n")
    output = tmp_path / "index"
    manifest = storage.build_index(kb, output, storage.defaults("bm25_sections"))
    assert manifest["sources"] == [origin]
    assert manifest["files"] == ["index.sqlite3"]
    assert not any("sha256" in key for key in manifest)
    with sqlite3.connect(output / "index.sqlite3") as db:
        assert [row[1] for row in db.execute("PRAGMA table_info(documents)")] == [
            "origin", "body"
        ]
    summary, records, _, _ = documents.write_catalog(kb, output)
    assert not any("sha256" in key for key in summary)
    assert not any("sha256" in key for key in records[0])
    (output / "index.sqlite3").write_bytes(b"changed artifact")
    assert storage.verify_artifacts(output) == manifest


@pytest.mark.parametrize("name", ["../outside", "nested/file", "missing"])
def test_artifact_inventory_preserves_path_and_presence_checks(tmp_path, name):
    (tmp_path / "build.json").write_text(json.dumps({"files": [name]}))
    with pytest.raises(ValueError):
        storage.verify_artifacts(tmp_path)


def test_snapshot_identity_changes_for_cache_without_source_checksums(tmp_path):
    origin = "a" * 64
    source = tmp_path / f"{origin}.md"
    source.write_text("first")
    _, sources, first = documents.snapshot(tmp_path)
    source.write_text("second")
    _, second_sources, second = documents.snapshot(tmp_path)
    assert sources == second_sources == [origin]
    assert first != second


def test_legacy_index_selection_uses_settings_and_paths(tmp_path, monkeypatch):
    config = storage.defaults("hybrid_bge")
    config["embedding_backend"] = "local"
    config["embedding_model"] = "BAAI/bge-m3"
    (tmp_path / "configs").mkdir()
    storage.write_json(tmp_path / "configs/hybrid_bge.local.json", config)
    lexical = tmp_path / "cache/corpus/lexical"
    lexical.mkdir(parents=True)
    storage.write_json(lexical / "build.json", {"config": config, "chunks": 1})
    matrix = tmp_path / "cache/embeddings/matrix"
    matrix.mkdir(parents=True)
    (matrix / "vectors.f32").write_bytes(b"\0" * 4096)
    monkeypatch.setattr(storage.embeddings, "model_identity", lambda _: {"model": "fixture"})
    result = storage.inspect_legacy_run(tmp_path)
    assert result["index"] == str(lexical)
    assert result["completed_rows"] == 1
