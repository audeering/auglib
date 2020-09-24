import pytest

import audb

import auglib


class Ones(auglib.Transform):
    def call(self, buf: auglib.AudioBuffer):
        buf.data.fill(1)


db = audb.load('testdata', version='1.5.0')
pytest.DATA_COLUMN = db['happiness.dev.gold']['happiness'].get()
pytest.DATA_FILES = db['files'].get()[:5]
pytest.DATA_TABLE = db['happiness.dev.gold'].get()
pytest.TRANSFORM_ONES = Ones()
