"""Compatibility import; the implementation is shared in retrieval/."""
import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "retrieval" / "__init__.py").is_file():
        sys.path.insert(0, str(_root))
        break
from retrieval import chunking as _api

sys.modules[__name__] = _api
