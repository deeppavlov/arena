"""Compatibility API. Algorithms live in the five retrieval stage modules."""

import socket as socket
import urllib as urllib

from . import search
from .chunking import split_document as split_document
from .config import BOUNDARY as BOUNDARY
from .config import DESIGNATION as DESIGNATION
from .config import GRAPH_QUERY as GRAPH_QUERY
from .config import ORIGIN as ORIGIN
from .config import SOURCE_LINK as SOURCE_LINK
from .config import STOP as STOP
from .config import WORDS as WORDS
from .config import defaults as defaults
from .config import validate_config as validate_config
from .documents import extract_links as extract_links
from .embeddings import Encoder as Encoder
from .embeddings import NoRedirect as NoRedirect
from .embeddings import post_json as post_json
from .embeddings import rerank as rerank
from .search import bm25 as bm25
from .search import cli as cli
from .search import dense as dense
from .search import expand_links as expand_links
from .search import extract_question as extract_question
from .search import retrieve as retrieve
from .search import rrf as rrf
from .search import terms as terms
from .search import validate_answer as validate_answer
from .storage import build_index as build_index
from .storage import digest as digest
from .storage import write_json as write_json


def answer_question(question, result, mode, timeout, *, validate=True):
    return search.answer_question(
        question, result, mode, timeout, validate=validate, request=post_json
    )
