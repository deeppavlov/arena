import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SEED = "sc71-first-20261019"
SPLITS = {"train": 2, "public": 6, "private": 12}
TOPICS = tuple(f"T{number}" for number in range(1, 6))
ORIGIN = re.compile(r"[0-9a-f]{64}")


class PreparationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def invalid_constant(value: str) -> None:
    raise PreparationError("nonfinite JSON value")


def parse_json(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as from_error:
        raise PreparationError("invalid UTF-8 JSON document") from from_error


def read_file(root: Path, name: str) -> bytes:
    relative = PurePosixPath(name)
    require(
        bool(name) and not relative.is_absolute() and relative.as_posix() == name
        and not any(part in {".", ".."} for part in relative.parts)
        and "\\" not in name and "\x00" not in name,
        "noncanonical source path",
    )
    path = root
    for part in relative.parts:
        path /= part
        require(not path.is_symlink(), "source symlink is forbidden")
    require(path.is_file(), "required source file is missing")
    try:
        return path.read_bytes()
    except OSError as error:
        raise PreparationError("source file cannot be read") from error


def text(data: bytes) -> str:
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PreparationError("source text is not UTF-8") from error
    require(bool(value.strip()), "source text is empty")
    return value


def load_corpus(kb_root: Path, expected_files: int) -> tuple[dict[str, Any], dict[str, str]]:
    index_bytes = read_file(kb_root, "index.tsv")
    manifest_bytes = read_file(kb_root, "manifest.json")
    manifest = parse_json(manifest_bytes)
    require(isinstance(manifest, dict), "invalid KB manifest")
    require(
        manifest.get("corpus_version") == "v1.1-raw"
        and manifest.get("snapshot_tag") == "snapshot-2026-09-08-full",
        "KB version is not the approved Paddle snapshot",
    )
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for line in text(index_bytes).splitlines():
        fields = line.split("\t")
        require(len(fields) == 4, "KB index must have four columns")
        identifier, size, original, digest = fields
        require(
            ORIGIN.fullmatch(identifier) is not None and identifier not in paths
            and size.isdigit() and bool(original) and ORIGIN.fullmatch(digest) is not None,
            "invalid KB index entry",
        )
        require(identifier == sha256(original.encode("utf-8")), "KB origin identity mismatch")
        paths[identifier] = original
        hashes[identifier] = sha256(read_file(kb_root, f"norm/{identifier}.md"))
        require(hashes[identifier] == digest, "KB Markdown checksum mismatch")
    norm_files = {path.name for path in (kb_root / "norm").iterdir()}
    require(
        len(paths) == expected_files
        and norm_files == {f"{identifier}.md" for identifier in paths},
        "KB corpus file count or membership mismatch",
    )
    objects = manifest.get("objects")
    require(isinstance(objects, list), "invalid KB original-object manifest")
    object_ids = set()
    for item in objects:
        require(isinstance(item, dict), "invalid KB original-object entry")
        identifier = item.get("origin_id")
        require(
            isinstance(identifier, str) and identifier not in object_ids
            and identifier in paths and isinstance(item.get("relative_posix"), str)
            and paths.get(identifier) == item.get("relative_posix"),
            "KB source identity mismatch",
        )
        object_ids.add(identifier)
    provenance = {
        "corpus_version": manifest["corpus_version"],
        "snapshot_tag": manifest["snapshot_tag"], "file_count": len(paths),
        "index_sha256": sha256(index_bytes), "manifest_sha256": sha256(manifest_bytes),
        "norm_sha256": sha256(encoded(hashes)),
    }
    archive = kb_root / "_kb/archive_members.jsonl"
    if archive.exists():
        provenance["archive_members_sha256"] = sha256(read_file(
            kb_root, "_kb/archive_members.jsonl",
        ))
    return provenance, paths


def check_gold(gold: dict[str, Any], paths: dict[str, str], kb_root: Path) -> None:
    cache: dict[str, str] = {}

    def document(identifier: Any) -> str:
        require(isinstance(identifier, str) and identifier in paths, "unknown gold document ID")
        if identifier not in cache:
            cache[identifier] = " ".join(text(read_file(
                kb_root, f"norm/{identifier}.md",
            )).split())
        return cache[identifier]

    def quote(identifier: Any, value: Any) -> None:
        require(isinstance(value, str) and len(value) >= 20, "invalid gold quote")
        require(" ".join(value.split()) in document(identifier), "gold quote differs from KB")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "origin_id" or key.endswith("_origin_id"):
                    document(item)
                elif key.endswith("_origin_ids"):
                    require(isinstance(item, list), "invalid gold document IDs")
                    for identifier in item:
                        document(identifier)
                visit(item)
            if "relative_posix" in value:
                require(
                    paths.get(str(value.get("origin_id"))) == value["relative_posix"],
                    "gold source path differs from KB index",
                )
            if "quote" in value:
                quote(value.get("origin_id"), value["quote"])
                for identifier in value.get("equivalent_origin_ids", []):
                    quote(identifier, value["quote"])
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(gold)
    if "t1" in gold:
        item = gold["t1"]
        require(isinstance(item, dict), "invalid T1 gold structure")
        old_id = item.get("superseded_origin_id") if item.get("mode") == "two_documents" else (
            item.get("current_origin_id")
        )
        quote(old_id, item.get("old_text"))
        quote(item.get("current_origin_id"), item.get("new_text"))


def mkdir(path: Path) -> None:
    if not path.exists():
        mkdir(path.parent)
        path.mkdir(mode=0o700)


def write_file(path: Path, data: bytes) -> None:
    mkdir(path.parent)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(data)


def prepare_dataset(
    source: Path, kb_root: Path, train_output: Path, closed_output: Path,
    seed: str = SEED, expected_kb_files: int = 4454,
) -> dict[str, Any]:
    require(bool(seed), "split seed must be nonempty")
    roots = [path.resolve() for path in (source, kb_root, train_output, closed_output)]
    for index, path in enumerate(roots):
        for other in roots[index + 1:]:
            require(
                not path.is_relative_to(other) and not other.is_relative_to(path),
                "source, KB and output directories must be separate",
            )
    for path in (train_output, closed_output):
        require(not path.exists() and not path.is_symlink(), "output directory already exists")
    source, kb_root, train_output, closed_output = roots
    manifest_bytes = read_file(source, "manifest.json")
    manifest = parse_json(manifest_bytes)
    require(isinstance(manifest, dict), "invalid source manifest")
    require(manifest.get("dataset_version") == "1", "source dataset version must be 1")
    require(manifest.get("task_prompt", "task.md") == "task.md", "unexpected task prompt path")
    rows = manifest.get("examples")
    require(isinstance(rows, list) and len(rows) == 100, "source must contain exactly 100 examples")
    expected_ids = {f"{topic}-{number:02}" for topic in TOPICS for number in range(1, 21)}
    require(all(isinstance(row, dict) for row in rows), "invalid source manifest example")
    require(
        {row.get("example_id") for row in rows} == expected_ids,
        "source example IDs must be exactly the approved 100 IDs",
    )
    task_bytes = read_file(source, "task.md")
    task = text(task_bytes)
    corpus, paths = load_corpus(kb_root, expected_kb_files)
    examples: dict[str, dict[str, Any]] = {}
    original_files: dict[str, dict[str, bytes]] = {}
    question_hashes: list[str] = []
    for row in rows:
        identifier = row["example_id"]
        topic = identifier[:2]
        sources = row.get("sources")
        question_path = f"materials/{identifier}/question.md"
        reference_path = f"ground_truth/{identifier}.md"
        require(
            isinstance(sources, list) and len(sources) == 1 and isinstance(sources[0], dict)
            and sources[0].get("path") == question_path
            and sources[0].get("kind") == "document"
            and row.get("ground_truth_file") == reference_path
            and row.get("ground_truth_origin") == "model_reviewed",
            "source manifest paths do not match the approved layout",
        )
        files = {name: read_file(source, name) for name in (
            question_path, reference_path, f"ground_truth/{identifier}.json",
        )}
        question = text(files[question_path])
        question_hashes.append(sha256(" ".join(question.split()).encode()))
        reference_markdown = text(files[reference_path])
        gold = parse_json(files[f"ground_truth/{identifier}.json"])
        require(
            isinstance(gold, dict) and gold.get("example_id") == identifier
            and gold.get("topic") == topic and gold.get("status") == "validated"
            and type(gold.get("answerable")) is bool and isinstance(gold.get("citations"), list)
            and isinstance(gold.get("must_include"), list)
            and isinstance(gold.get("must_not_include"), list),
            "gold identity, status or required fields are invalid",
        )
        check_gold(gold, paths, kb_root)
        reference = encoded({
            "format": "sc71-reference-v1", "topic": topic,
            "reference_markdown": reference_markdown, "gold": gold,
        }).decode()
        input_text = task + "\n\n" + question
        require(len(input_text) <= 500000 and len(reference) <= 500000, "example exceeds limits")
        examples[identifier] = {"id": identifier, "input": input_text, "reference": reference}
        original_files[identifier] = files
    duplicates = sum(count > 1 for count in Counter(question_hashes).values())
    require(duplicates == 0, f"duplicate normalized question groups: {duplicates}")
    membership: dict[str, list[str]] = {split: [] for split in SPLITS}
    for topic in TOPICS:
        identifiers = sorted(
            (identifier for identifier in examples if identifier.startswith(topic + "-")),
            key=lambda identifier: (sha256(f"{seed}\0{identifier}".encode()), identifier),
        )
        offset = 0
        for split, count in SPLITS.items():
            membership[split].extend(identifiers[offset:offset + count])
            offset += count
    report: dict[str, Any] = {
        "format": "sc71-prepared-dataset-v1", "seed": seed,
        "algorithm": "sha256(seed + NUL + example_id), 2/6/12 per topic",
        "dataset_version": manifest.get("dataset_version"),
        "source_manifest_sha256": sha256(manifest_bytes), "kb": corpus,
        "counts": {split: len(identifiers) for split, identifiers in membership.items()},
        "topic_counts": {
            split: {topic: count for topic in TOPICS} for split, count in SPLITS.items()
        }, "splits": {},
    }
    full_manifest: dict[str, Any] = {"summary": report, "membership": membership}
    for split, identifiers in membership.items():
        output = train_output if split == "train" else closed_output / split
        selected = [examples[identifier] for identifier in sorted(identifiers)]
        dataset_bytes = b"".join(encoded(example) + b"\n" for example in selected)
        pairs = sorted((example["input"], example["reference"]) for example in selected)
        split_manifest = {
            "format": "sc71-split-v1", "split": split, "seed": seed,
            "dataset_sha256": sha256(dataset_bytes), "dataset_digest": sha256(encoded(pairs)),
            "source_files": {"task.md": sha256(task_bytes)},
            "examples": sorted(identifiers), "kb": corpus,
        }
        write_file(output / "dataset.jsonl", dataset_bytes)
        write_file(output / "source/task.md", task_bytes)
        for identifier in sorted(identifiers):
            for name, data in original_files[identifier].items():
                write_file(output / "source" / name, data)
                split_manifest["source_files"][name] = sha256(data)
        split_manifest_bytes = encoded(split_manifest) + b"\n"
        write_file(output / "manifest.json", split_manifest_bytes)
        report["splits"][split] = {
            "dataset_sha256": split_manifest["dataset_sha256"],
            "dataset_digest": split_manifest["dataset_digest"],
            "manifest_sha256": sha256(split_manifest_bytes),
        }
    write_file(closed_output / "source-manifest.json", manifest_bytes)
    write_file(closed_output / "manifest.json", encoded(full_manifest) + b"\n")
    write_file(train_output / "summary.json", encoded(report) + b"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SC-71 splits inside the operator boundary",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kb-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--closed-output", type=Path, required=True)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    try:
        report = prepare_dataset(
            args.source, args.kb_root, args.train_output, args.closed_output, args.seed,
        )
    except PreparationError as error:
        parser.exit(1, f"preparation failed: {error}\n")
    except (OSError, ValueError, TypeError, KeyError):
        parser.exit(1, "preparation failed: invalid source metadata or filesystem operation\n")
    print(encoded(report).decode())


if __name__ == "__main__":
    main()
