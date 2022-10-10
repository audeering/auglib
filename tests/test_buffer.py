import re
import os

import numpy as np
import pytest

import audiofile
import auglib
import audobject


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


def test_resolver(tmpdir):

    with auglib.AudioBuffer(4, 8, unit='samples') as aux:

        error_msg = (
            "Cannot serialize an instance of "
            "<class 'auglib.core.buffer.AudioBuffer'>. "
        )
        with pytest.raises(ValueError, match=error_msg):
            auglib.transform.Mix(aux).to_yaml_s()

        error_msg = (
            "Cannot serialize list if it contains an instance of "
            "<class 'auglib.core.buffer.AudioBuffer'>. "
        )
        with pytest.raises(ValueError, match=error_msg):
            list_with_aux = auglib.observe.List([aux])
            auglib.transform.Mix(list_with_aux).to_yaml_s()

        aux_path = os.path.join(tmpdir, 'aux.wav')
        aux.write(aux_path)

        transform = auglib.transform.Mix(aux_path)
        yaml_s = transform.to_yaml_s()
        assert audobject.from_yaml_s(yaml_s) == transform

        list_with_aux_path = auglib.observe.List([aux_path])
        transform = auglib.transform.Mix(list_with_aux_path)
        yaml_s = transform.to_yaml_s()
        assert audobject.from_yaml_s(yaml_s) == transform


@pytest.mark.parametrize('sampling_rate', [-10, -10.0, 0, 0.0, 3.4, 10.0])
@pytest.mark.parametrize('duration', [1])
@pytest.mark.parametrize('unit', ['seconds', 'samples'])
def test_sampling_rate_error(sampling_rate, duration, unit):

    error_msg = (
        'Sampling rate must be an integer and greater than zero, '
        f'not {sampling_rate} Hz'
    )
    with pytest.raises(ValueError, match=error_msg):
        auglib.AudioBuffer(duration, sampling_rate, unit=unit)


def test_shape_error(tmpdir):

    signal = np.zeros((2, 8))
    sampling_rate = 8000
    path = os.path.join(tmpdir, 'stereo.wav')
    audiofile.write(path, signal, sampling_rate)

    with pytest.raises(ValueError):
        auglib.AudioBuffer.from_array(signal, sampling_rate)

    with pytest.raises(ValueError):
        auglib.AudioBuffer.read(path)


@pytest.mark.parametrize(
    'duration, unit, sampling_rate',
    [
        (0, 'samples', 8000),
        (0, 'seconds', 8000),
        (-1, 'samples', 8000),
        (-1, 'seconds', 8000),
        (0.0001, 'seconds', 8000),
    ],
)
def test_size_error(tmpdir, duration, unit, sampling_rate):

    error_msg = (
        'Empty buffers are not supported '
        f"(duration: {duration} {unit}, sampling rate: {sampling_rate} Hz)"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        auglib.AudioBuffer(duration, sampling_rate, unit=unit)


def test_size_error_from_file_or_array(tmpdir):

    signal = np.array([[]])
    sampling_rate = 8000
    path = os.path.join(tmpdir, 'empty.wav')
    audiofile.write(path, signal, sampling_rate)

    error_msg = (
        'Empty buffers are not supported '
        f"(duration: 0 samples, sampling rate: {sampling_rate} Hz"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        auglib.AudioBuffer.from_array(signal, sampling_rate)
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        auglib.AudioBuffer.read(path)


def test_str():

    with auglib.AudioBuffer(4, 8000, unit='samples') as buf:
        assert str(buf) == str(buf.to_array())
