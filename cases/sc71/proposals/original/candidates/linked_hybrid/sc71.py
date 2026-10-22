"""Compatibility import; the implementation is shared in retrieval/."""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

for _root in Path(__file__).resolve().parents:
    if (_root / "retrieval" / "__init__.py").is_file():
        sys.path.insert(0, str(_root))
        break
from retrieval import api as _api

if TYPE_CHECKING:
    from retrieval.api import * 

sys.modules[__name__] = _api
