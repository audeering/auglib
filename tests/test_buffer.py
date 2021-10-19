import os
import numpy as np
import pytest

from auglib import AudioBuffer
from auglib.utils import to_samples


@pytest.mark.parametrize('dur,sr,unit',
                         [(1.0, 8000, None),
                          (8000, 1600, 'samples'),
                          (1000, 44100, 'milliseconds')])
def test_init(dur, sr, unit):

    unit = unit or 'seconds'
    n = to_samples(dur, sampling_rate=sr, unit=unit)
    with AudioBuffer(dur, sr, unit=unit) as buf:
        assert len(buf) == n
        np.testing.assert_equal(buf._data, np.zeros(n))
        buf._data += 1
        np.testing.assert_equal(buf._data, np.ones(n))
    assert buf._obj is None
    assert buf._data is None


@pytest.mark.parametrize('dur,sr',
                         [(1.0, 8000),
                          (1.0, 1600),
                          (1.0, 44100)])
def test_file(dur, sr):

    path = './test.wav'
    n = to_samples(dur, sampling_rate=sr)
    x = np.random.random(n).astype(np.float32)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.write(path)
    with AudioBuffer.read(path) as buf:
        np.testing.assert_almost_equal(buf._data, x, decimal=3)

    os.remove(path)


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_from_array(n, sr):

    x = np.random.random(n).astype(np.float32)
    with AudioBuffer.from_array(x, sr) as buf:
        np.testing.assert_equal(x, buf._data)
