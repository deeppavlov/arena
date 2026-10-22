"""Prepare corpus assets once, outside question answering. No dataset is read."""

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from retrieval import storage
from retrieval.common import load_env

METHODS = (
    "bm25_sections",
    "hybrid_bge",
    "linked_hybrid",
    "structure_bm25",
    "planned_bm25",
    "file_agent",
    "amendment_bm25",
)


def main():
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=(*METHODS, "all"))
    parser.add_argument("--kb-dir", type=Path)
    parser.add_argument("--indexes-dir", type=Path, default=ROOT / "indexes")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--devices", nargs="+", help="logical devices after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--resume-from", type=Path, help="verified prefix checkpoint directory")
    parser.add_argument("--progress", action="store_true", help="JSON progress on stderr")
    parser.add_argument("--inspect-run", type=Path, help="read-only inspection of the original run")
    parser.add_argument("--capture-run", type=Path, help="capture a paused original writer")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--verify", type=Path, metavar="INDEX")
    args = parser.parse_args()
    try:
        if args.inspect_run or args.capture_run:
            if args.inspect_run and args.capture_run:
                parser.error("choose --inspect-run or --capture-run")
            if args.capture_run:
                if not args.checkpoint_dir:
                    parser.error("--capture-run requires --checkpoint-dir")
                result = storage.capture_legacy_prefix(args.capture_run, args.checkpoint_dir)
            else:
                result = storage.inspect_legacy_run(args.inspect_run)
            print(json.dumps(result, ensure_ascii=False))
            return
        if args.verify:
            manifest = storage.verify_artifacts(args.verify)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "documents": manifest.get("documents"),
                        "chunks": manifest["chunks"],
                    }
                )
            )
            return
        if not args.candidate or not args.kb_dir:
            parser.error("--candidate and --kb-dir are required for preparation")
        names = list(METHODS) if "all" in args.candidate else list(dict.fromkeys(args.candidate))
        if len(names) != 1 and (args.build_dir or args.settings):
            parser.error("--build-dir and --settings apply to one candidate only")
        if args.env_file:
            load_env(args.env_file)
        for name in names:
            output = args.build_dir or args.indexes_dir / name
            manifest = storage.prepare(
                args.kb_dir,
                output,
                name,
                settings_path=args.settings,
                cache_dir=args.cache_dir or args.indexes_dir / ".prepared",
                embedding_revision=args.embedding_revision,
                source_manifest=args.source_manifest,
                resume_from=args.resume_from,
                devices=args.devices,
                progress=args.progress,
            )
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "build_dir": str(output),
                        "reused": manifest["reused"],
                        "documents": manifest["documents"],
                        "chunks": manifest["chunks"],
                        "dimension": manifest["dimension"],
                        "preparation": manifest["preparation"],
                    },
                    ensure_ascii=False,
                )
            )
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
