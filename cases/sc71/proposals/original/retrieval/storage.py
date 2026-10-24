"""Preparation artifacts and lexical index building."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import embeddings
from .chunking import split_document
from .config import (
    ORIGIN,
    defaults,
    validate_config,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def implementation():
    root = Path(__file__).parent
    return fingerprint({p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.py"))})


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def verify_artifacts(output, filename="build.json"):
    output = Path(output)
    manifest = json.loads((output / filename).read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact manifest has no file inventory")
    for name in files:
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("unsafe artifact path")
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact file is missing or not regular")
    if "vectors.f32" in files:
        dimension, chunks = manifest.get("dimension"), manifest.get("chunks")
        if type(dimension) is not int or dimension <= 0 or type(chunks) is not int:
            raise ValueError("invalid vector dimensions")
        if (output / "vectors.f32").stat().st_size != chunks * dimension * 4:
            raise ValueError("vector shape mismatch")
    return manifest


@contextmanager
def preparation_lock(target):
    """Fail explicitly on concurrent/stale preparation; never remove another process's lock."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            f"preparation already locked: {lock.name}; inspect the owner process"
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started": time.time()}, stream)
        yield
    finally:
        lock.unlink()


def publish_artifact(target, builder):
    """Promote a completed sibling to a ready directory."""
    stage = target.with_name(".incomplete-" + target.name + "-" + uuid.uuid4().hex)
    builder(stage)
    verify_artifacts(stage)
    if target.exists():
        if any(target.iterdir()):
            raise ValueError("preparation destination became nonempty")
        target.rmdir()  # Only the explicitly selected, verified empty directory.
    stage.rename(target)


def load_candidate(name):
    allowed = {
        "bm25_sections",
        "hybrid_bge",
        "linked_hybrid",
        "structure_bm25",
        "planned_bm25",
        "file_agent",
        "amendment_bm25",
    }
    if name not in allowed:
        raise ValueError("unknown candidate")
    return importlib.import_module("candidates." + name + ".candidate")


def _legacy_writer_state(pid, root):
    """Linux-only read-only check. This function never signals a process."""
    proc = Path("/proc") / str(pid)
    if not sys.platform.startswith("linux"):
        raise ValueError("legacy writer inspection requires Linux /proc")
    try:
        status = (proc / "status").read_text()
    except FileNotFoundError:
        return "not_running"
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    uid = int(fields["Uid"].split()[0])
    get_uid = getattr(os, "getuid", None)
    if get_uid is None:
        raise ValueError("Linux user identity is unavailable")
    current_uid = get_uid()
    if uid != current_uid:
        raise ValueError("legacy writer belongs to another user")
    state = fields["State"].split()[0]
    if state == "Z":
        return "not_running"
    command = (proc / "cmdline").read_bytes().split(b"\0")
    allowed = {b"scripts/prepare.py", str(root / "scripts/prepare.py").encode()}
    if (proc / "cwd").resolve() != root or not any(arg in allowed for arg in command):
        raise ValueError("lock PID is not this run's preparation writer")
    return state


def inspect_legacy_run(run):
    """Locate a unique legacy index and matrix with matching settings."""
    run = Path(run).resolve()
    config = json.loads((run / "configs/hybrid_bge.local.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config["method"] != "hybrid_bge" or config["embedding_backend"] != "local":
        raise ValueError("legacy capture supports the original local hybrid_bge run")
    if config["embedding_model"] != "BAAI/bge-m3":
        raise ValueError("legacy capture requires BGE-M3 with dimension 1024")
    identity = embeddings.model_identity(config)
    candidates = []
    for build in (run / "cache/corpus").glob("*/build.json"):
        meta = json.loads(build.read_text(encoding="utf-8"))
        settings = meta.get("config", {})
        if all(settings.get(key) == config[key] for key in ("chunk_chars", "overlap_lines")):
            candidates.append((build.parent, meta))
    if len(candidates) != 1:
        raise ValueError("cannot uniquely identify the original completed lexical index")
    index, meta = candidates[0]
    parent = run / "cache/embeddings"
    matches = list(parent.glob("*/vectors.f32"))
    if len(matches) != 1:
        raise ValueError("cannot uniquely identify the original matrix")
    vectors = matches[0]
    matrix_key = vectors.parent.name
    if matrix_key.startswith(".incomplete-"):
        matrix_key = matrix_key.removeprefix(".incomplete-").rsplit("-", 1)[0]
    lock = parent / (matrix_key + ".lock")
    pid = json.loads(lock.read_text())["pid"] if lock.exists() else None
    state = _legacy_writer_state(pid, run) if pid else "not_running"
    size = vectors.stat().st_size
    rows = size // (1024 * 4)
    if rows > meta["chunks"] or vectors.is_symlink():
        raise ValueError("invalid legacy matrix size or type")
    return {
        "run": str(run), "index": str(index), "vectors": str(vectors),
        "writer_pid": pid, "writer_state": state,
        "completed_rows": rows, "chunks": meta["chunks"],
        "omitted_tail_bytes": size % (1024 * 4),
        "config": config, "embedding_identity": identity,
    }


def capture_legacy_prefix(run, output):
    """Copy complete vectors from a paused/dead known writer; preserve its files/locks."""
    import numpy as np

    details = inspect_legacy_run(run)
    if details["writer_state"] not in {"T", "t", "not_running"}:
        raise ValueError("pause the exact legacy writer first with kill -STOP PID")
    index, source, output = Path(details["index"]), Path(details["vectors"]), Path(output)
    verify_artifacts(index)
    if output.exists():
        raise ValueError("checkpoint destination already exists; choose a new directory")
    if not details["completed_rows"]:
        raise ValueError("no complete legacy vectors have been written yet")
    before = source.stat()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(".incomplete-capture-" + uuid.uuid4().hex)
    stage.mkdir()
    remaining = details["completed_rows"]
    with source.open("rb") as reader, (stage / "vectors.f32").open("wb") as writer:
        while remaining:
            rows = min(remaining, 4096)
            raw = reader.read(rows * 1024 * 4)
            if len(raw) != rows * 1024 * 4:
                raise ValueError("legacy matrix changed or was truncated")
            vectors = np.frombuffer(raw, dtype="<f4").reshape(rows, 1024)
            if not np.isfinite(vectors).all() or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1, atol=1e-4
            ):
                raise ValueError("legacy matrix contains invalid/non-normalized vectors")
            writer.write(raw)
            remaining -= rows
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("legacy matrix changed during capture; no checkpoint published")
    if details["writer_pid"] and _legacy_writer_state(
        details["writer_pid"], Path(details["run"])
    ) not in {"T", "t", "not_running"}:
        raise ValueError("legacy writer resumed during capture")
    metadata = embeddings.prefix_metadata(
        index, details["config"], details["completed_rows"], 1024
    )
    if metadata["embedding_identity"] != details["embedding_identity"]:
        raise ValueError("model changed during capture")
    metadata["legacy_import"] = {
        "omitted_tail_bytes": details["omitted_tail_bytes"],
        "writer_pid": details["writer_pid"],
    }
    write_json(stage / "checkpoint.json", metadata)
    embeddings.read_prefix(stage, metadata)
    stage.rename(output)
    return metadata


def prepare(
    kb,
    output,
    name,
    *,
    settings_path=None,
    cache_dir=None,
    embedding_revision=None,
    source_manifest=None,
    resume_from=None,
    devices=None,
    progress=False,
):
    """Prepare immutable candidate assets. This function is never called by search/run."""
    from . import common, documents
    from .structural import StructuralBackend

    started = time.monotonic()
    if progress:
        print(json.dumps({"phase": "corpus", "status": "checking"}), file=sys.stderr, flush=True)
    kb, _, corpus_key = documents.snapshot(kb)
    output = Path(output).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else output.parent / ".prepared"
    for target in (output, cache):
        if target == kb or kb in target.parents or target in kb.parents:
            raise ValueError("preparation must be outside the corpus")
    old_style = name in {"bm25_sections", "hybrid_bge", "linked_hybrid"}
    candidate = (
        SimpleNamespace(PACKAGE=Path(__file__).resolve().parents[1] / "candidates" / name)
        if old_style
        else load_candidate(name)
    )
    path = Path(settings_path) if settings_path else candidate.PACKAGE / "settings.json"
    value = (
        defaults(name)
        if old_style and settings_path is None and not path.exists()
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if old_style:
        validate_config(value)
        if value["method"] != name:
            raise ValueError("candidate settings method mismatch")
    else:
        candidate.validate(value)
    config = value if old_style else value["lexical"]
    embeddings.validate_devices(config, devices)
    if (resume_from or devices) and name not in {"hybrid_bge", "linked_hybrid"}:
        raise ValueError("resume/devices are supported only for embedding hybrids")
    structured = name in {"structure_bm25", "file_agent", "amendment_bm25"}
    source_info = (
        documents.source_metadata(kb, source_manifest)[1]
        if name
        in {
            "file_agent",
            "amendment_bm25",
        }
        else None
    )
    identity = (
        embeddings.model_identity(config, embedding_revision)
        if name
        in {
            "hybrid_bge",
            "linked_hybrid",
        }
        else None
    )
    code = implementation()
    package_code = (
        {}
        if old_style
        else {p.name: p.read_text(encoding="utf-8") for p in sorted(candidate.PACKAGE.glob("*.py"))}
    )
    if (candidate.PACKAGE / "AGENT.md").exists():
        package_code["AGENT.md"] = (candidate.PACKAGE / "AGENT.md").read_text(encoding="utf-8")
    preparation_key = fingerprint(
        {
            "schema": 1,
            "corpus": corpus_key,
            "settings": value,
            "encoder": identity,
            "metadata": source_info,
            "implementation": code,
            "candidate": package_code,
        }
    )
    with preparation_lock(output):
        if (output / "build.json").exists():
            existing = verify_artifacts(output)
            if existing.get("preparation", {}).get("key") != preparation_key:
                raise ValueError("existing assets are incompatible; select a new --build-dir")
            return dict(existing, reused=True)
        if output.exists() and any(output.iterdir()):
            raise ValueError("incomplete/nonempty output; inspect it and choose a new --build-dir")
        lexical_key = fingerprint(
            {
                "corpus": corpus_key,
                "mode": "structure" if structured else "sections",
                "chunk_chars": config["chunk_chars"],
                "overlap_lines": config["overlap_lines"],
                "implementation": code,
            }
        )
        lexical = cache / "corpus" / lexical_key
        with preparation_lock(lexical):
            lexical_reused = (lexical / "build.json").exists()
            if lexical_reused:
                base_manifest = verify_artifacts(lexical)
            else:
                if progress:
                    print(
                        json.dumps({"phase": "lexical_index", "status": "building"}),
                        file=sys.stderr, flush=True,
                    )

                def build_lexical(stage):
                    if structured:
                        base_value = {
                            **value,
                            "candidate": "structure_bm25",
                        }
                        StructuralBackend("structure_bm25", candidate.PACKAGE).build(
                            kb, stage, base_value
                        )
                    else:
                        build_index(kb, stage, dict(config, method="bm25_sections", rerank=False))

                publish_artifact(lexical, build_lexical)
                base_manifest = verify_artifacts(lexical)
        matrix, matrix_manifest, matrix_reused = None, None, None
        if identity is not None:
            if progress:
                print(
                    json.dumps({"phase": "embeddings", "status": "checking_cache_or_prefix"}),
                    file=sys.stderr, flush=True,
                )
            matrix_key = fingerprint(
                {
                    "lexical_key": lexical_key,
                    "encoder": identity,
                    "batch": config["embedding_batch"],
                }
            )
            matrix = cache / "embeddings" / matrix_key
            with preparation_lock(matrix):
                matrix_reused = (matrix / "build.json").exists()
                if not matrix_reused:

                    def build_matrix(stage):
                        stage.mkdir()
                        info = embeddings.prepare_matrix(
                            lexical, stage, config, resume_from=resume_from,
                            devices=devices, progress=progress,
                        )
                        info.update(
                            embedding_identity=identity,
                            files=["checkpoint.json", "vectors.f32"],
                        )
                        write_json(stage / "build.json", info)

                    publish_artifact(matrix, build_matrix)
                matrix_manifest = verify_artifacts(matrix)

        def assemble(stage):
            shutil.copytree(lexical, stage)
            manifest = json.loads(json.dumps(base_manifest))
            manifest["config"] = config
            if old_style:
                for key in (
                    "candidate",
                    "candidate_settings",
                    "implementation",
                    "shared_implementation",
                ):
                    manifest.pop(key, None)
                manifest.update(edges=0, link_diagnostics={"ambiguous": 0, "unresolved": 0})
                if matrix is not None:
                    assert matrix_manifest is not None
                    shutil.copy2(matrix / "vectors.f32", stage / "vectors.f32")
                    manifest.update(
                        dimension=matrix_manifest["dimension"], embedding_identity=identity,
                        embedding_execution=matrix_manifest.get("execution"),
                    )
                    if (matrix / "checkpoint.json").exists():
                        shutil.copy2(matrix / "checkpoint.json", stage / "checkpoint.json")
                if name == "linked_hybrid":
                    db = sqlite3.connect(stage / "index.sqlite3")
                    try:
                        edges, diagnostics = documents.extract_links(
                            list(db.execute("SELECT origin,body FROM documents"))
                        )
                        db.executemany(
                            "INSERT INTO edges VALUES (?,?,?,?,?)",
                            [
                                (e["source"], e["target"], e["quote"], e["offset"], e["type"])
                                for e in edges
                            ],
                        )
                        db.commit()
                    finally:
                        db.close()
                    manifest.update(edges=len(edges), link_diagnostics=diagnostics)
            elif name in {"file_agent", "amendment_bm25"}:
                manifest = candidate.finish_preparation(kb, stage, manifest, value, source_manifest)
            else:
                manifest = common.seal_build(stage, manifest, value, candidate.PACKAGE)
            if (
                source_info is not None
                and documents.source_metadata(kb, source_manifest)[1] != source_info
            ):
                raise ValueError("source metadata changed during preparation")
            manifest.update(
                build_seconds=time.monotonic() - started,
                preparation={
                    "schema_version": 1,
                    "key": preparation_key,
                    "corpus_key": lexical_key,
                    "lexical_reused": lexical_reused,
                    "embeddings_reused": matrix_reused,
                },
            )
            manifest["files"] = sorted(
                p.name for p in stage.iterdir() if p.is_file() and p.name != "build.json"
            )
            write_json(stage / "build.json", manifest)

        publish_artifact(output, assemble)
        return dict(verify_artifacts(output), reused=False)


def build_index(kb: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    from .documents import extract_links
    from .search import terms as _terms

    validate_config(config)
    kb = kb.resolve()
    if (kb / "norm").is_dir():
        kb = kb / "norm"
    if not kb.is_dir():
        raise ValueError("KB directory does not exist")
    output = output.resolve()
    if output == kb or kb in output.parents or output in kb.parents:
        raise ValueError("build directory must be separate from corpus")
    if output.exists() and any(output.iterdir()):
        raise ValueError("build directory must be empty; use a new build version")
    paths = sorted(kb.glob("*.md"))
    if not paths or any(not ORIGIN.fullmatch(p.stem) or p.is_symlink() for p in paths):
        raise ValueError("KB must contain only regular Markdown with 64-hex origin names")
    encoder = embeddings.Encoder(config) if config["method"] != "bm25_sections" else None
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    db = sqlite3.connect(output / "index.sqlite3")
    try:
        db.executescript("""
        CREATE TABLE documents(origin TEXT PRIMARY KEY, body TEXT);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, origin TEXT, seq INTEGER, section TEXT,
          start INTEGER, end INTEGER, line_start INTEGER, line_end INTEGER,
          body TEXT, length INTEGER);
        CREATE INDEX chunk_document ON chunks(origin, seq);
        CREATE TABLE postings(term TEXT, chunk INTEGER, tf INTEGER, PRIMARY KEY(term, chunk));
        CREATE TABLE vocabulary(term TEXT PRIMARY KEY, df INTEGER);
        CREATE TABLE edges(source TEXT, target TEXT, quote TEXT, offset INTEGER, type TEXT);
        CREATE INDEX edge_source ON edges(source);
        CREATE INDEX edge_target ON edges(target);
        """)
        sources = []
        chunk_id = 0
        for path in paths:
            raw = path.read_bytes()
            body = raw.decode("utf-8-sig")
            if "\x00" in body:
                raise ValueError(f"NUL bytes in document {path.name}")
            sources.append(path.stem)
            db.execute("INSERT INTO documents VALUES (?, ?)", (path.stem, body))
            for seq, chunk in enumerate(
                split_document(body, config["chunk_chars"], config["overlap_lines"])
            ):
                counted = Counter(_terms(chunk["text"]))
                db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        path.stem,
                        seq,
                        chunk["section"],
                        chunk["start"],
                        chunk["end"],
                        chunk["line_start"],
                        chunk["line_end"],
                        chunk["text"],
                        sum(counted.values()),
                    ),
                )
                db.executemany(
                    "INSERT INTO postings VALUES (?, ?, ?)",
                    [(term, chunk_id, tf) for term, tf in counted.items()],
                )
                chunk_id += 1
            db.commit()
        if not chunk_id:
            raise ValueError("corpus contains no text")
        db.execute("INSERT INTO vocabulary SELECT term, COUNT(*) FROM postings GROUP BY term")
        link_stats = {"ambiguous": 0, "unresolved": 0}
        edge_count = 0
        if config["method"] == "linked_hybrid":
            edges, link_stats = extract_links(
                list(db.execute("SELECT origin, body FROM documents"))
            )
            db.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                [(e["source"], e["target"], e["quote"], e["offset"], e["type"]) for e in edges],
            )
            edge_count = len(edges)
        db.commit()
        dimension = None
        if encoder:
            import numpy as np

            batch = config["embedding_batch"]
            with (output / "vectors.f32").open("wb") as stream:
                for start in range(0, chunk_id, batch):
                    rows = db.execute(
                        "SELECT c.body, substr(d.body,1,400) FROM chunks c JOIN documents d"
                        " ON c.origin=d.origin WHERE c.id>=? AND c.id<? ORDER BY c.id",
                        (start, start + batch),
                    )
                    texts = [
                        (header[:400] + "\n" + body)[: config["embedding_max_chars"]]
                        for body, header in rows
                    ]
                    vectors = encoder.encode(texts)
                    if dimension is None:
                        dimension = vectors.shape[1]
                    if vectors.shape[1] != dimension:
                        raise ValueError("embedding dimension changed within build")
                    stream.write(np.asarray(vectors, dtype="<f4").tobytes())
        mean_length = db.execute("SELECT AVG(length) FROM chunks").fetchone()[0]
    finally:
        db.close()
    manifest = {
        "schema_version": 1,
        "config": config,
        "documents": len(paths),
        "chunks": chunk_id,
        "sources": sources,
        "dimension": dimension,
        "mean_length": mean_length,
        "edges": edge_count,
        "link_diagnostics": link_stats,
        "build_seconds": time.monotonic() - started,
        "python": sys.version.split()[0],
        "files": sorted(p.name for p in output.iterdir() if p.is_file()),
    }
    write_json(output / "build.json", manifest)
    return manifest
