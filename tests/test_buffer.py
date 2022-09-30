import os

import numpy as np
import pytest

import audiofile
import auglib


@pytest.mark.parametrize(
    'dur,sr',
    [
        (1.0, 8000),
        (1.0, 1600),
        (1.0, 44100),
    ],
)
def test_file(tmpdir, dur, sr):

    path = 'test.wav'
    n = auglib.utils.to_samples(dur, sampling_rate=sr)
    x = np.random.random(n).astype(np.float32)

    with auglib.AudioBuffer.from_array(x, sr) as buf:
        buf.write(path, root=tmpdir, sampling_rate=48000)
        assert audiofile.sampling_rate(os.path.join(tmpdir, path)) == 48000
        buf.write(path, root=tmpdir)
        assert audiofile.sampling_rate(os.path.join(tmpdir, path)) == sr
    with auglib.AudioBuffer.read(path, root=tmpdir) as buf:
        np.testing.assert_almost_equal(buf._data, x, decimal=3)


@pytest.mark.parametrize(
    'n,sr',
    [
        (10, 8000),
        (10, 44100),
    ],
)
def test_from_array(n, sr):

    x = np.random.random(n).astype(np.float32)
    with auglib.AudioBuffer.from_array(x, sr) as buf:
        np.testing.assert_equal(x, buf._data)


@pytest.mark.parametrize(
    'dur,sr,unit',
    [
        (1.0, 8000, None),
        (8000, 1600, 'samples'),
        (1000, 44100, 'milliseconds'),
    ],
)
def test_init(dur, sr, unit):

    unit = unit or 'seconds'
    n = auglib.utils.to_samples(dur, sampling_rate=sr, unit=unit)
    with auglib.AudioBuffer(dur, sr, unit=unit) as buf:
        assert len(buf) == n
        assert buf.duration == n / sr
        np.testing.assert_equal(buf._data, np.zeros(n))
        buf._data += 1
        np.testing.assert_equal(buf._data, np.ones(n))
    assert buf._obj is None
    assert buf._data is None


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('signal', [[0, 1, 2, 3]])
def test_rms(sampling_rate, signal):
    with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf:
        expected_rms = np.sqrt(np.mean(np.square(signal)))
        expected_rms_db = 20 * np.log10(expected_rms)
        np.testing.assert_almost_equal(
            buf.rms,
            expected_rms,
            decimal=4,
        )
        np.testing.assert_almost_equal(
            buf.rms_db,
            expected_rms_db,
            decimal=4,
        )


def test_shape_error(tmpdir):

    signal = np.zeros((2, 8))
    sampling_rate = 8000
    path = os.path.join(tmpdir, 'stereo.wav')
    audiofile.write(path, signal, sampling_rate)

    with pytest.raises(ValueError):
        auglib.AudioBuffer.from_array(signal, sampling_rate)

    with pytest.raises(ValueError):
        auglib.AudioBuffer.read(path)


def test_str():

    with auglib.AudioBuffer(4, -8000, unit='samples') as buf:
        assert str(buf) == str(buf.to_array())
