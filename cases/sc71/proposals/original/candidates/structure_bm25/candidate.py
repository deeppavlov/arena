"""structure_bm25: unchanged structural retrieval strategy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval import api, common
from retrieval.chunking import parse_document as parse_document
from retrieval.structural import StructuralBackend
from retrieval.structural import ranges_for as ranges_for

base = api

NAME = "structure_bm25"
PACKAGE = Path(__file__).resolve().parent
_backend = StructuralBackend(NAME, PACKAGE)
defaults = _backend.defaults
validate = _backend.validate
build = _backend.build
retrieve = _backend.retrieve
def cli(stage):
    common.cli(stage, PACKAGE, NAME, build, retrieve)
