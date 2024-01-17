import numpy as np
import pytest

import audb

import auglib


class Ones(auglib.transform.Base):
    def _call(self, base: np.ndarray, *, sampling_rate: int = None):
        return np.ones(base.shape)


db = audb.load(
    "emodb",
    version="1.4.1",
    media=["wav/03a01Fa.wav", "wav/03a01Nc.wav", "wav/03a01Wa.wav"],
    verbose=False,
)
pytest.AUDB_ROOT = audb.default_cache_root()
pytest.DATA_FILES = db.files
pytest.TRANSFORM_ONES = Ones()
