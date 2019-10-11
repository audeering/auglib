import os
import numpy as np
from scipy import signal
import pytest

from auglib import AudioBuffer
from auglib.utils import dur_to_samples


@pytest.mark.parametrize('dur,sr,unit',
                         [(1.0, 8000, None),
                          (8000, 1600, 'samples'),
                          (1000, 44100, 'milliseconds')])
def test_init(dur, sr, unit):

    unit = unit or 'seconds'
    n = dur_to_samples(dur, sr, unit=unit)
    with AudioBuffer(dur, sr, unit=unit) as buf:
        assert len(buf) == n
        np.testing.assert_equal(buf.data, np.zeros(n))
        buf.data += 1
        np.testing.assert_equal(buf.data, np.ones(n))
    assert buf.obj is None
    assert buf.data is None


@pytest.mark.parametrize('dur,sr',
                         [(1.0, 8000),
                          (1.0, 1600),
                          (1.0, 44100)])
def test_file(dur, sr):

    path = './test.wav'
    n = dur_to_samples(dur, sr)
    x = np.random.random(n).astype(np.float32)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.to_file(path)
    with AudioBuffer.from_file(path) as buf:
        np.testing.assert_almost_equal(buf.data, x, decimal=3)

    os.remove(path)


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_from_array(n, sr):

    x = np.random.random(n).astype(np.float32)
    with AudioBuffer.from_array(x, sr) as buf:
        np.testing.assert_equal(x, buf.data)


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_normalize(n, sr):

    with AudioBuffer.from_array(np.random.uniform(-0.5, 0.5, n), sr) as buf:
        buf.normalize_by_peak()
        assert np.isclose(np.abs(buf.data).max(), 1.0)


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_clip(n, sr):

    x = np.c_[np.random.uniform(-1.5, 1.0, n // 2),
              np.random.uniform(1.0, 1.5, n - (n // 2))]
    np.random.shuffle(x)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.clip()
        np.isclose(np.abs(buf.data).max(), 1.0)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.clip(threshold=0.5)
        np.isclose(np.abs(buf.data).max(), 0.5)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.clip(threshold=0.5, normalize=True)
        assert np.isclose(np.abs(buf.data).max(), 1.0)

    x = np.random.uniform(1, 2, n)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.clip_by_ratio(0.5, normalize=False)
        assert buf.data.max() <= np.median(x)

    with AudioBuffer.from_array(x, sr) as buf:
        buf.clip_by_ratio(0.5, normalize=True)
        assert np.isclose(np.abs(buf.data).max(), 1.0)


@pytest.mark.parametrize('n,sr,clip',
                         [(10, 8000, False),
                          (10, 44100, True)])
def test_gain_stage(n, sr, clip):

    x = np.random.uniform(-0.1, 1.0, n)
    with AudioBuffer.from_array(x, sr) as buf:
        buf.gain_stage(20.0, clip=clip)
        if clip:
            assert np.abs(buf.data).max() <= 1.0
        else:
            assert np.isclose(np.abs(buf.data).max(), 10.0 * np.abs(x).max())


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_fft_convolve(n, sr):

    with AudioBuffer(n, sr, unit='samples') as base:
        with AudioBuffer(n, sr, unit='samples') as aux:
            base.data[0] = 1
            aux.data += 1
            base.fft_convolve(aux, keep_tail=False)
            np.testing.assert_equal(base.data, np.ones(len(base)))


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_filter(n, sr):
    sig_in = np.ones(n * sr, dtype='float32')

    b, a = signal.butter(1, 2.0 / 4, 'lowpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer(n, sr) as buf:
        buf.data += 1
        buf.low_pass(sr / 4)
        np.testing.assert_almost_equal(buf.data, sig_out)

    b, a = signal.butter(1, 2.0 / 4, 'highpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer(n, sr) as buf:
        buf.data += 1
        buf.high_pass(sr / 4)
        np.testing.assert_almost_equal(buf.data, sig_out)

    b, a = signal.butter(1, (2.0 / 8) * np.array([1, 3]), 'bandpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer(n, sr) as buf:
        buf.data += 1
        buf.band_pass(sr // 4, sr // 4)
        np.testing.assert_almost_equal(buf.data, sig_out)
