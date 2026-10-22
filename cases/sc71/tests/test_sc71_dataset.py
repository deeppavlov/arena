import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cases.sc71.tools.prepare_dataset import PreparationError, prepare_dataset
from evaluator.contracts import Example

ORIGIN_ID = hashlib.sha256(b"original.pdf").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def source_package(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    kb = tmp_path / "kb"
    (source / "ground_truth").mkdir(parents=True)
    (source / "validation_evidence").mkdir()
    (source / "validation_evidence" / "secret.md").write_text("do not export")
    (kb / "norm").mkdir(parents=True)
    origin_id = ORIGIN_ID
    text = "Синтетический нормативный текст для проверки цитаты.\r\n".encode()
    (kb / "norm" / f"{origin_id}.md").write_bytes(text)
    (kb / "index.tsv").write_text(
        f"{origin_id}\t10\toriginal.pdf\t{hashlib.sha256(text).hexdigest()}\n"
    )
    write_json(kb / "manifest.json", {
        "corpus_version": "v1.1-raw", "snapshot_tag": "snapshot-2026-09-08-full",
        "objects": [{"origin_id": origin_id, "relative_posix": "original.pdf"}],
    })
    (source / "task.md").write_bytes("Общая инструкция.\r\n".encode())
    examples = []
    for topic in range(1, 6):
        for number in range(1, 21):
            identifier = f"T{topic}-{number:02}"
            question_dir = source / "materials" / identifier
            question_dir.mkdir(parents=True)
            (question_dir / "question.md").write_bytes(f"Вопрос {identifier}?\r\n".encode())
            (source / "ground_truth" / f"{identifier}.md").write_bytes(
                f"Эталон {identifier}.\r\n".encode()
            )
            write_json(source / "ground_truth" / f"{identifier}.json", {
                "example_id": identifier, "topic": f"T{topic}", "answerable": True,
                "status": "validated", "citations": [{
                    "origin_id": origin_id, "relative_posix": "original.pdf",
                    "quote": text.decode().strip(), "role": "primary",
                }], "must_include": ["синтетический"], "must_not_include": [],
            })
            examples.append({
                "example_id": identifier,
                "sources": [{
                    "path": f"materials/{identifier}/question.md", "kind": "document",
                    "title": "ВОПРОС",
                }],
                "ground_truth_file": f"ground_truth/{identifier}.md",
                "ground_truth_origin": "model_reviewed",
            })
    write_json(source / "manifest.json", {
        "dataset_version": "1", "task_prompt": "task.md", "examples": examples,
    })
    return source, kb


def prepare(source: Path, kb: Path, output: Path) -> dict:
    return prepare_dataset(source, kb, output / "train", output / "closed", expected_kb_files=1)


def test_split_is_complete_deterministic_and_preserves_authoritative_bytes(
    source_package: tuple[Path, Path], tmp_path: Path,
) -> None:
    source, kb = source_package
    first = tmp_path / "first"
    report = prepare(source, kb, first)
    assert report["counts"] == {"train": 10, "public": 30, "private": 60}
    assert report["topic_counts"] == {
        split: {f"T{topic}": count for topic in range(1, 6)}
        for split, count in (("train", 2), ("public", 6), ("private", 12))
    }
    all_ids = set()
    for split in ("train", "public", "private"):
        directory = first / ("train" if split == "train" else f"closed/{split}")
        data = (directory / "dataset.jsonl").read_bytes()
        examples = [Example.model_validate_json(line) for line in data.splitlines()]
        identifiers = {example.id for example in examples}
        assert not all_ids & identifiers
        all_ids |= identifiers
        manifest = json.loads((directory / "manifest.json").read_bytes())
        assert manifest["dataset_sha256"] == hashlib.sha256(data).hexdigest()
        canonical = sorted((example.input, example.reference) for example in examples)
        assert manifest["dataset_digest"] == hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        for example in examples:
            reference = json.loads(example.reference or "")
            assert reference["format"] == "sc71-reference-v1"
            assert reference["topic"] == example.id[:2]
            assert reference["reference_markdown"] == f"Эталон {example.id}.\r\n"
            assert example.input == f"Общая инструкция.\r\n\n\nВопрос {example.id}?\r\n"
            for relative in (
                f"materials/{example.id}/question.md",
                f"ground_truth/{example.id}.md", f"ground_truth/{example.id}.json",
            ):
                assert (directory / "source" / relative).read_bytes() == (
                    source / relative
                ).read_bytes()
        for path in directory.rglob("*"):
            assert "validation_evidence" not in str(path)
            assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert len(all_ids) == 100
    serialized_report = json.dumps(report)
    assert all(identifier not in serialized_report for identifier in all_ids)
    assert "Эталон" not in serialized_report
    manifest = json.loads((source / "manifest.json").read_bytes())
    manifest["examples"].reverse()
    write_json(source / "manifest.json", manifest)
    second = tmp_path / "second"
    prepare(source, kb, second)
    for relative in (
        "train/dataset.jsonl", "closed/public/dataset.jsonl", "closed/private/dataset.jsonl",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "traversal", "topic", "symlink", "version", "origin",
])
def test_invalid_source_is_rejected_before_output(
    source_package: tuple[Path, Path], tmp_path: Path, change: str,
) -> None:
    source, kb = source_package
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if change == "missing":
        manifest["examples"].pop()
    elif change == "duplicate":
        manifest["examples"][-1] = manifest["examples"][0]
    elif change == "traversal":
        manifest["examples"][0]["sources"][0]["path"] = "../outside.md"
    elif change == "topic":
        path = source / "ground_truth/T1-01.json"
        gold = json.loads(path.read_bytes())
        gold["topic"] = "T2"
        write_json(path, gold)
    elif change == "version":
        manifest["dataset_version"] = "2"
    elif change == "origin":
        manifest["examples"][0]["ground_truth_origin"] = "unreviewed"
    else:
        path = source / "materials/T1-01/question.md"
        path.unlink()
        path.symlink_to(source / "materials/T1-02/question.md")
    write_json(manifest_path, manifest)
    output = tmp_path / "result"
    with pytest.raises(PreparationError):
        prepare(source, kb, output)
    assert not output.exists()


@pytest.mark.parametrize("change", [
    "hash", "quote", "origin", "duplicate_question", "malformed_t1", "kb_manifest",
    "index_identity",
])
def test_corpus_checks_and_question_deduplication_do_not_disclose_content(
    source_package: tuple[Path, Path], tmp_path: Path, change: str,
) -> None:
    source, kb = source_package
    if change == "hash":
        (kb / "norm" / f"{ORIGIN_ID}.md").write_text("SENSITIVE CORPUS CHANGE")
    elif change == "index_identity":
        fake_id = "a" * 64
        (kb / "norm" / f"{ORIGIN_ID}.md").rename(kb / "norm" / f"{fake_id}.md")
        for name in ("index.tsv", "manifest.json"):
            path = kb / name
            path.write_text(path.read_text().replace(ORIGIN_ID, fake_id))
        for path in (source / "ground_truth").glob("*.json"):
            path.write_text(path.read_text().replace(ORIGIN_ID, fake_id))
    elif change == "duplicate_question":
        (source / "materials/T1-02/question.md").write_bytes(
            (source / "materials/T1-01/question.md").read_bytes()
        )
    elif change == "kb_manifest":
        path = kb / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["objects"] = [{"origin_id": "z"}]
        write_json(path, manifest)
    else:
        path = source / "ground_truth/T1-01.json"
        gold = json.loads(path.read_bytes())
        if change == "malformed_t1":
            gold["t1"] = "SENSITIVE GOLD CHANGE"
        else:
            key = "quote" if change == "quote" else "origin_id"
            gold["citations"][0][key] = "SENSITIVE GOLD CHANGE"
        write_json(path, gold)
    with pytest.raises(PreparationError) as failure:
        prepare(source, kb, tmp_path / "result")
    assert "SENSITIVE" not in str(failure.value)
    assert "T1-01" not in str(failure.value)
    assert not (tmp_path / "result").exists()


def test_existing_and_overlapping_outputs_are_rejected(
    source_package: tuple[Path, Path], tmp_path: Path,
) -> None:
    source, kb = source_package
    output = tmp_path / "result"
    prepare(source, kb, output)
    before = (output / "train/dataset.jsonl").read_bytes()
    with pytest.raises(PreparationError):
        prepare(source, kb, output)
    assert (output / "train/dataset.jsonl").read_bytes() == before
    with pytest.raises(PreparationError):
        prepare_dataset(source, kb, source / "train", tmp_path / "closed", expected_kb_files=1)
    with pytest.raises(PreparationError):
        prepare_dataset(source, kb, tmp_path / "nested/train", tmp_path / "nested")


def test_cli_failure_is_safe_and_does_not_emit_traceback(tmp_path: Path) -> None:
    result = subprocess.run([
        sys.executable, "cases/sc71/tools/prepare_dataset.py", "--source", str(tmp_path),
        "--kb-root", str(tmp_path / "kb"), "--train-output", str(tmp_path / "train"),
        "--closed-output", str(tmp_path / "closed"),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
