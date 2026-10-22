"""Compatibility CLI delegating to the shared runtime."""
import sys
from pathlib import Path

for root in Path(__file__).resolve().parents:
    if (root / "retrieval" / "__init__.py").is_file():
        sys.path.insert(0, str(root))
        break
from retrieval.runtime import main as _main

import candidate


def main(stage):
    return _main(stage, candidate)
