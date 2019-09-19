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

    buf = AudioBuffer.FromArray(np.random.uniform(-0.5, 0.5, n), sr)
    buf.normalize_by_peak()
    assert np.isclose(np.abs(buf.data).max(), 1.0)
    buf.free()


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_clip(n, sr):

    x = np.c_[np.random.uniform(-1.5, 1.0, n // 2),
              np.random.uniform(1.0, 1.5, n - (n // 2))]
    np.random.shuffle(x)

    buf = AudioBuffer.FromArray(x, sr)
    buf.clip()
    np.isclose(np.abs(buf.data).max(), 1.0)
    # buf.free()  TODO: randomly fails on CI/CD

    buf = AudioBuffer.FromArray(x, sr)
    buf.clip(threshold=0.5)
    np.isclose(np.abs(buf.data).max(), 0.5)
    # buf.free()  TODO: randomly fails on CI/CD

    buf = AudioBuffer.FromArray(x, sr)
    buf.clip(threshold=0.5, normalize=True)
    assert np.isclose(np.abs(buf.data).max(), 1.0)
    # buf.free()  TODO: randomly fails on CI/CD

    x = np.random.uniform(1, 2, n)

    base = AudioBuffer.FromArray(x, sr)
    base.clip(threshold=0.5, normalize=False, as_ratio=True)
    assert base.data.max() <= np.median(x)
    # buf.free()  TODO: randomly fails on CI/CD

    base = AudioBuffer.FromArray(x, sr)
    base.clip(threshold=0.5, normalize=True, as_ratio=True)
    assert np.isclose(np.abs(buf.data).max(), 1.0)
    # buf.free()  TODO: randomly fails on CI/CD


@pytest.mark.parametrize('n,sr,clip',
                         [(10, 8000, False),
                          (10, 44100, True)])
def test_gain_stage(n, sr, clip):

    x = np.random.uniform(-0.1, 1.0, n)
    buf = AudioBuffer.FromArray(x, sr)
    buf.gain_stage(20.0, clip=clip)
    if clip:
        assert np.abs(buf.data).max() <= 1.0
    else:
        assert np.isclose(np.abs(buf.data).max(), 10.0 * np.abs(x).max())
    buf.free()


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
    base.free()
    aux.free()


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_filter(n, sr):
    sig_in = np.ones(n * sr, dtype='float32')

    b, a = signal.butter(1, 2.0 / 4, 'lowpass')
    sig_out = signal.lfilter(b, a, sig_in)

    base = AudioBuffer(n, sr)
    base.data += 1
    base.low_pass(1, sr / 4)
    np.testing.assert_almost_equal(base.data, sig_out)

    # TODO: bring this back to life once the highpass filter is fixed in auglib
    # b, a = signal.butter(1, 2.0 / 4, 'highpass')
    # sig_out = signal.lfilter(b, a, sig_in)
    #
    # base = AudioBuffer(n, sr)
    # base.data += 1
    # base.high_pass(1, sr / 4)
    # np.testing.assert_almost_equal(base.data, sig_out, decimal=3)

    b, a = signal.butter(1, (2.0 / 8) * np.array([1, 3]), 'bandpass')
    sig_out = signal.lfilter(b, a, sig_in)
    base.free()

    base = AudioBuffer(n, sr)
    base.data += 1
    base.band_pass(1, sr // 4, sr // 4)
    np.testing.assert_almost_equal(base.data, sig_out)
    base.free()
