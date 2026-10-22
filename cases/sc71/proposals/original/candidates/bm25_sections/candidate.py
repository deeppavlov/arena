"""bm25_sections: strategy identity; the ranking implementation is shared."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval import api as base

NAME = "bm25_sections"
PACKAGE = Path(__file__).resolve().parent

def defaults():
    return base.defaults(NAME)

def retrieve(output, question):
    return base.retrieve(output, question)
