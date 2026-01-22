from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE
import os

import pytest
from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.rest import CodeBlockParser
from sybil.parsers.rest import DocTestParser


@pytest.fixture(scope="module")
def execute_in_tmpdir(tmpdir_factory):
    path = tmpdir_factory.mktemp("doctest")
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(cwd)


pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=NORMALIZE_WHITESPACE | ELLIPSIS),
        CodeBlockParser(language="python", evaluator=PythonEvaluator()),
    ],
    patterns=["*.rst"],
    fixtures=["execute_in_tmpdir"],
).pytest()
