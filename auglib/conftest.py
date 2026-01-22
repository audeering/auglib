from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser


pytest_collect_file = Sybil(
    parsers=[DocTestParser(optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)],
    patterns=["*.py"],
    excludes=["core/transform.py"],
).pytest()
