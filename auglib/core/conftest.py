import pytest

import auglib


@pytest.fixture(scope="package", autouse=True)
def prepare_docstring_tests(doctest_namespace):
    doctest_namespace["seed"] = auglib.seed
