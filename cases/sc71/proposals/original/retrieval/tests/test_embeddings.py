import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    monkeypatch.setenv("SC71_EMBED_REVISION", "synthetic-v1")
    from retrieval import embeddings

    return embeddings


@pytest.fixture
def config():
    return {
        "embedding_backend": "http",
        "embedding_model": "synthetic-model",
        "embedding_max_chars": 1024,
        "embedding_batch": 1,
    }


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "index"
    path.mkdir()
    (path / "build.json").write_text(json.dumps({"chunks": 2, "sources": ["source-a"]}))
    with sqlite3.connect(path / "index.sqlite3") as db:
        db.execute("CREATE TABLE documents (origin TEXT, body TEXT)")
        db.execute("CREATE TABLE chunks (id INTEGER, origin TEXT, body TEXT)")
        db.execute("INSERT INTO documents VALUES ('source-a', 'Synthetic header')")
        db.executemany(
            "INSERT INTO chunks VALUES (?, 'source-a', ?)",
            [(0, "Synthetic first row"), (1, "Synthetic second row")],
        )
    return path


def checkpoint(path, metadata, vectors):
    path.mkdir()
    (path / "checkpoint.json").write_text(json.dumps(metadata))
    (path / "vectors.f32").write_bytes(np.asarray(vectors, dtype="<f4").tobytes())
    return path


def test_prefix_metadata_uses_sources_and_encoding_settings(api, config, index):
    metadata = api.prefix_metadata(index, config, 1, 2)
    assert metadata == {
        "schema": "sc71-embedding-prefix-v1",
        "chunks": 2,
        "sources": ["source-a"],
        "embedding_identity": api.model_identity(config),
        "completed_rows": 1,
        "dimension": 2,
        "batch_per_device": 1,
    }


def test_local_model_identity_uses_directory_without_reading_weights(
    api, config, tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"synthetic weights")
    monkeypatch.setenv("SC71_LOCAL_EMBED_MODEL", str(model))
    config["embedding_backend"] = "local"
    before = api.model_identity(config)
    weights.write_bytes(b"updated synthetic weights")
    assert api.model_identity(config) == before
    assert before["model_path"] == str(model.resolve())
    assert set(before) == {
        "backend", "model", "max_chars", "max_seq_length", "normalize", "dtype", "prefix",
        "model_path",
    }


def test_empty_local_model_directory_is_rejected(api, config, tmp_path, monkeypatch):
    monkeypatch.setenv("SC71_LOCAL_EMBED_MODEL", str(tmp_path))
    config["embedding_backend"] = "local"
    with pytest.raises(ValueError, match="empty"):
        api.model_identity(config)


def test_prefix_accepts_valid_replacement_vectors(api, config, index, tmp_path):
    metadata = api.prefix_metadata(index, config, 1, 2)
    source = checkpoint(tmp_path / "checkpoint", metadata, [[1, 0], [0, 1]])
    assert api.read_prefix(source, metadata) == metadata
    (source / "vectors.f32").write_bytes(np.asarray([[0, 1], [np.nan, 0]], dtype="<f4").tobytes())
    assert api.read_prefix(source, metadata) == metadata


@pytest.mark.parametrize("values", [[[0, 0]], [[np.nan, 0]], [[2, 0]]])
def test_prefix_rejects_invalid_vectors(api, config, index, tmp_path, values):
    metadata = api.prefix_metadata(index, config, 1, 2)
    source = checkpoint(tmp_path / "checkpoint", metadata, values)
    with pytest.raises(ValueError, match="invalid/non-normalized"):
        api.read_prefix(source, metadata)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("sources", ["source-b"], "sources differs"),
        ("completed_rows", 3, "row count"),
        ("completed_rows", True, "row count"),
        ("dimension", 0, "dimension"),
    ],
)
def test_prefix_rejects_invalid_metadata(api, config, index, tmp_path, field, value, error):
    expected = api.prefix_metadata(index, config, 1, 2)
    metadata = dict(expected, **{field: value})
    source = checkpoint(tmp_path / "checkpoint", metadata, [[1, 0]])
    with pytest.raises(ValueError, match=error):
        api.read_prefix(source, expected)


def test_prefix_rejects_truncated_vectors(api, config, index, tmp_path):
    metadata = api.prefix_metadata(index, config, 2, 2)
    source = checkpoint(tmp_path / "checkpoint", metadata, [[1, 0]])
    with pytest.raises(ValueError, match="truncated"):
        api.read_prefix(source, metadata)


def test_matrix_resumes_completed_rows_with_structural_metadata(
    api, config, index, tmp_path, monkeypatch
):
    class Encoder:
        def __init__(self, config):
            self.config = config

        def encode(self, texts):
            assert len(texts) == 1 and "second row" in texts[0]
            return np.asarray([[0, 1]], dtype=np.float32)

    monkeypatch.setattr(api, "Encoder", Encoder)
    metadata = api.prefix_metadata(index, config, 1, 2)
    source = checkpoint(tmp_path / "checkpoint", metadata, [[1, 0]])
    output = tmp_path / "output"
    output.mkdir()
    result = api.prepare_matrix(index, output, config, resume_from=source)
    assert result["chunks"] == 2
    assert result["dimension"] == 2
    assert result["execution"]["resumed_rows"] == 1
    assert set(result["execution"]) == {
        "devices", "batch_per_device", "resumed_rows", "source_provenance", "seconds",
    }
    assert json.loads((output / "checkpoint.json").read_text()) == dict(metadata, completed_rows=2)
    vectors = np.fromfile(output / "vectors.f32", dtype="<f4").reshape(2, 2)
    np.testing.assert_array_equal(vectors, [[1, 0], [0, 1]])
