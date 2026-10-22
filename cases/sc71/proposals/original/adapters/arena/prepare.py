"""Prepare context for ONE Arena question; never build corpus assets."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
for root in Path(__file__).resolve().parents:
    if (root / "retrieval" / "__init__.py").is_file():
        sys.path.insert(0, str(root))
        break

from adapters.arena.runtime import main

if __name__ == "__main__":
    main()
