import numpy as np
from scipy import signal
import pytest

from auglib import AudioBuffer, lib
from auglib.utils import random_seed
from auglib.transform import Mix, Append, NormalizeByPeak, Clip, ClipByRatio, \
    GainStage, FFTConvolve, LowPass, HighPass, BandPass, WhiteNoiseUniform,\
    WhiteNoiseGaussian, PinkNoise, Tone, ToneShape
from auglib.utils import to_samples, to_db


@pytest.mark.parametrize('base_dur,aux_dur,sr,unit',
                         [(1.0, 1.0, 8000, None),
                          (16000, 8000, 16000, 'samples'),
                          (500, 1000, 44100, 'ms')])
def test_mix(base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = to_samples(base_dur, sr, unit=unit)
    n_aux = to_samples(aux_dur, sr, unit=unit)

    n_min = min(n_base, n_aux)
    n_max = max(n_base, n_aux)

    # init auxiliary buffer

    aux = AudioBuffer(aux_dur, sr, value=1.0, unit=unit)

    # default mix

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux)(base)
        assert np.sum(np.abs(base.data[n_min:])) == 0
        np.testing.assert_equal(base.data[:n_min], aux.data[:n_min])

    # loop auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, loop_aux=True)(base)
        np.testing.assert_equal(base.data, np.ones(n_base))

    # extend base

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, extend_base=True)(base)
        assert len(base) == n_max
        np.testing.assert_equal(base.data[:n_aux], np.ones(n_aux))

    # restrict length of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, read_dur_aux=1, unit='samples')(base)
        assert base.data[0] == 1 and np.sum(np.abs(base.data[1:])) == 0

    # read position of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, read_pos_aux=n_aux - 1, unit='samples')(base)
        assert base.data[0] == 1 and np.sum(np.abs(base.data[1:])) == 0

    # write position of base

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, write_pos_base=n_base - 1, unit='samples')(base)
        assert base.data[-1] == 1 and np.sum(np.abs(base.data[:-1])) == 0

    # set gain of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, gain_aux_db=to_db(2), loop_aux=True)(base)
        assert all(base.data == 2)
        Mix(aux, gain_base_db=to_db(0.5), loop_aux=True)(base)
        assert all(base.data == 2)

    # clipping

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, gain_aux_db=to_db(2), loop_aux=True, clip_mix=True)(base)
        assert all(base.data == 1)

    aux.free()


@pytest.mark.parametrize('base_dur,aux_dur,sr,unit',
                         [(1.0, 1.0, 8000, None),
                          (16000, 8000, 16000, 'samples'),
                          (500, 1000, 44100, 'ms')])
def test_append(base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = to_samples(base_dur, sr, unit=unit)
    n_aux = to_samples(aux_dur, sr, unit=unit)

    aux = AudioBuffer(aux_dur, sr, unit=unit, value=1.0)

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Append(aux)(base)
        np.testing.assert_equal(base.data[:n_base], np.zeros(n_base))
        np.testing.assert_equal(base.data[n_base:], np.ones(n_aux))

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Append(aux, read_pos_aux=n_aux - 1, unit='samples')(base)
        np.testing.assert_equal(base.data[:n_base], np.zeros(n_base))
        assert len(base.data) == n_base + 1
        assert base.data[-1] == 1

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Append(aux, read_dur_aux=1, unit='samples')(base)
        np.testing.assert_equal(base.data[:n_base], np.zeros(n_base))
        assert len(base.data) == n_base + 1
        assert base.data[-1] == 1

    aux.free()


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_normalize(n, sr):

    with AudioBuffer.from_array(np.linspace(-0.5, 0.5, num=n), sr) as buf:
        NormalizeByPeak()(buf)
        assert np.abs(buf.data).max() == 1.0


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_clip(n, sr):

    x = np.c_[np.random.uniform(-1.5, 1.0, n // 2),
              np.random.uniform(1.0, 1.5, n - (n // 2))]
    np.random.shuffle(x)

    with AudioBuffer.from_array(x, sr) as buf:
        Clip()(buf)
        np.isclose(np.abs(buf.data).max(), 1.0)

    with AudioBuffer.from_array(x, sr) as buf:
        Clip(threshold=0.5)(buf)
        np.isclose(np.abs(buf.data).max(), 0.5)

    with AudioBuffer.from_array(x, sr) as buf:
        Clip(threshold=0.5, normalize=True)(buf)
        assert np.isclose(np.abs(buf.data).max(), 1.0)

    x = np.random.uniform(1, 2, n)

    with AudioBuffer.from_array(x, sr) as buf:
        ClipByRatio(0.5, normalize=False)(buf)
        assert buf.data.max() <= np.median(x)

    with AudioBuffer.from_array(x, sr) as buf:
        ClipByRatio(0.5, normalize=True)(buf)
        assert np.isclose(np.abs(buf.data).max(), 1.0)
        assert np.abs(buf.data).min() >= 1. / 1.5


@pytest.mark.parametrize('n,sr,clip',
                         [(10, 8000, False),
                          (10, 44100, True)])
def test_gain_stage(n, sr, clip):

    x = np.random.uniform(-0.1, 1.0, n)
    with AudioBuffer.from_array(x, sr) as buf:
        GainStage(20.0, clip=clip)(buf)
        if clip:
            assert np.abs(buf.data).max() <= 1.0
        else:
            assert np.isclose(np.abs(buf.data).max(), 10.0 * np.abs(x).max())


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_fft_convolve(n, sr):

    with AudioBuffer(n, sr, unit='samples') as base:
        with AudioBuffer(n, sr, unit='samples', value=1.0) as aux:
            base.data[0] = 1
            FFTConvolve(aux, keep_tail=False)(base)
            np.testing.assert_equal(base.data, np.ones(len(base)))


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_filter(n, sr):
    sig_in = np.ones(n * sr, dtype='float32')

    b, a = signal.butter(1, 2.0 / 4, 'lowpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer(n, sr, value=1.0) as buf:
        LowPass(sr / 4)(buf)
        np.testing.assert_almost_equal(buf.data, sig_out)

    b, a = signal.butter(1, 2.0 / 4, 'highpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer(n, sr, value=1.0) as buf:
        HighPass(sr / 4)(buf)
        np.testing.assert_almost_equal(buf.data, sig_out)

    b, a = signal.butter(1, (2.0 / 8) * np.array([1, 3]), 'bandpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer(n, sr, value=1.0) as buf:
        BandPass(sr // 4, sr // 4)(buf)
        np.testing.assert_almost_equal(buf.data, sig_out)


@pytest.mark.parametrize('dur,sr,gain,seed',
                         [(1.0, 8000, 0.0, 1)])
def test_WhiteNoiseUniform(dur, sr, gain, seed):

    random_seed(seed)
    with WhiteNoiseUniform(gain_db=gain)(AudioBuffer(dur, sr)) as noise:
        with AudioBuffer(len(noise), noise.sampling_rate,
                         unit='samples') as buf:
            random_seed(seed)
            lib.AudioBuffer_addWhiteNoiseUniform(buf.obj, gain)
            np.testing.assert_equal(noise.data, buf.data)
            np.testing.assert_almost_equal(buf.data.mean(), 0, decimal=1)


@pytest.mark.parametrize('dur,sr,gain,stddev,seed',
                         [(1.0, 8000, 0.0, 0.3, 1)])
def test_WhiteNoiseGaussian(dur, sr, gain, stddev, seed):

    random_seed(seed)
    with WhiteNoiseGaussian(stddev=stddev, gain_db=gain)(AudioBuffer(dur, sr))\
            as noise:
        with AudioBuffer(len(noise), noise.sampling_rate,
                         unit='samples') as buf:
            random_seed(seed)
            lib.AudioBuffer_addWhiteNoiseGaussian(buf.obj, gain, stddev)
            np.testing.assert_equal(noise.data, buf.data)
            np.testing.assert_almost_equal(buf.data.mean(), 0, decimal=1)
            np.testing.assert_almost_equal(buf.data.std(), stddev, decimal=1)


@pytest.mark.parametrize('dur,sr,gain,seed',
                         [(1.0, 8000, 0.0, 1)])
def test_PinkNoise(dur, sr, gain, seed):

    random_seed(seed)
    with PinkNoise(gain_db=gain)(AudioBuffer(dur, sr)) as noise:
        with AudioBuffer(len(noise), noise.sampling_rate,
                         unit='samples') as buf:
            random_seed(seed)
            lib.AudioBuffer_addPinkNoise(buf.obj, gain)
            np.testing.assert_equal(noise.data, buf.data)
            np.testing.assert_almost_equal(buf.data.mean(), 0, decimal=1)


@pytest.mark.parametrize('freq', [1, 440])
def test_sine(freq):

    sr = 8000
    n = sr

    with Tone(freq, shape=ToneShape.SINE)(AudioBuffer(n, sr, unit='samples'))\
            as tone:
        sine = np.sin((np.arange(n, dtype=np.float) / sr) * 2 * np.pi * freq)
        np.testing.assert_almost_equal(tone.data, sine, decimal=3)
