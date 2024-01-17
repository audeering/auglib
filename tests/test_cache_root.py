import os

import pytest

import audeer

import auglib


def test_cache_root(tmpdir):
    auglib.config.CACHE_ROOT = tmpdir

    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
    )
    process_root = os.path.join(tmpdir, process.short_id)
    result = process.augment(pytest.DATA_FILES)
    result[0][0].startswith(str(tmpdir))

    assert auglib.default_cache_root() == tmpdir
    assert auglib.default_cache_root(process) == process_root
    assert len(audeer.list_file_names(process_root)) > 0

    auglib.clear_default_cache_root(process)
    assert os.path.exists(auglib.default_cache_root())
    assert not os.path.exists(process_root)

    auglib.clear_default_cache_root()
    assert os.path.exists(auglib.default_cache_root())
