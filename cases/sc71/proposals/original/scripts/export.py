"""Export source-only Arena candidates, including a frozen copy of the shared library."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from retrieval.storage import file_digest, load_candidate


def export(name, output, index=None, embedding_model=None):
    package = load_candidate(name).PACKAGE
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("export requires a new or empty directory")
    for source in (ROOT / "retrieval", ROOT / "candidates", ROOT / "adapters"):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("export must be separate from source directories")
    output.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "retrieval", ROOT / "adapters/arena", package):
        relative = source.relative_to(ROOT)
        destination = output / relative
        destination.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt", ".lock"}:
                shutil.copy2(path, destination / path.name)
    shutil.copy2(ROOT / "adapters/arena/prepare.py", output / "prepare.py")
    shutil.copy2(ROOT / "adapters/arena/prompt.md", output / "prompt.md")
    for filename in ("README.md", "METHODOLOGY.md"):
        shutil.copy2(ROOT / filename, output / filename)
    selection = {"candidate": name, "index": index or "indexes/" + name}
    if embedding_model:
        selection["embedding_model"] = embedding_model
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    manifest = {
        "arena_contract_commit": "4858f0bf02c3766b84d8dd4bce7810390b9028bc",
        "files": {
            p.relative_to(output).as_posix(): file_digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", help="Index path relative to /assets")
    parser.add_argument("--embedding-model", help="Local weights path relative to /assets")
    args = parser.parse_args()
    manifest = export(args.candidate, args.output, args.index, args.embedding_model)
    print(json.dumps({"files": len(manifest["files"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
