import numpy as np
import pytest

from auglib import AudioBuffer
from auglib.utils import dur2samples


@pytest.mark.parametrize('dur,sr,unit',
                         [(1.0, 8000, None),
                          (8000, 1600, 'samples'),

                          (1000, 44100, 'milliseconds')])
def test_init(dur, sr, unit):

    unit = unit or 'seconds'
    n = dur2samples(dur, sr, unit=unit)
    buf = AudioBuffer(dur, sr, unit=unit)
    assert len(buf) == n
    np.testing.assert_equal(buf.data, np.zeros(n))
    buf.data += 1
    np.testing.assert_equal(buf.data, np.ones(n))
    buf.free()


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_fromarray(n, sr):

    x = np.random.random(n).astype(np.float32)
    buf = AudioBuffer.FromArray(x, sr)
    np.testing.assert_equal(x, buf.data)
    buf.free()


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_normalize(n, sr):

    buf = AudioBuffer.FromArray(np.random.random(n), sr)
    buf.normalize()
    np.testing.assert_almost_equal([np.abs(buf.data).max()], [1.0])


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_fft_convolve(n, sr):

    base = AudioBuffer(n, sr, unit='samples')
    base.data[0] = 1
    aux = AudioBuffer(n, sr, unit='samples')
    aux.data += 1
    base.fft_convolve(aux, keep_tail=False)
    np.testing.assert_equal(base.data, np.ones(len(base)))
