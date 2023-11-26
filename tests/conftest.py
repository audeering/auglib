import numpy as np
import pytest

import audb

import auglib


class Ones(auglib.transform.Base):
    def _call(self, base: np.ndarray, *, sampling_rate: int = None):
        return np.ones(base.shape)


# Add access to audEERING data repos
audb.config.REPOSITORIES = [
    audb.Repository(
        name='data-public-local',
        host='https://artifactory.audeering.com/artifactory',
        backend='artifactory',
    )
]

db = audb.load('testdata', version='1.5.0')
pytest.AUDB_ROOT = audb.default_cache_root()
pytest.DATA_COLUMN = db['happiness.dev.gold']['happiness'].get()
pytest.DATA_FILES = db.files[:5]
pytest.DATA_TABLE = db['happiness.dev.gold'].get()
pytest.TRANSFORM_ONES = Ones()
