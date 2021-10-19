import numpy as np
from scipy import signal
import pytest

import auglib
from auglib import AudioBuffer
from auglib.core.buffer import lib
from auglib.core.exception import LibraryException
from auglib.transform import Mix, Append, AppendValue, Trim, NormalizeByPeak, \
    Clip, ClipByRatio, GainStage, FFTConvolve, LowPass, HighPass, BandPass, \
    BandStop, WhiteNoiseUniform, WhiteNoiseGaussian, PinkNoise, Tone, \
    CompressDynamicRange, AMRNB, Function
from auglib.utils import to_samples, to_db, from_db


@pytest.mark.parametrize('base_dur,aux_dur,sr,unit',
                         [(1.0, 1.0, 8000, None),
                          (16000, 8000, 16000, 'samples'),
                          (500, 1000, 44100, 'ms')])
def test_mix(base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = to_samples(base_dur, sampling_rate=sr, unit=unit)
    n_aux = to_samples(aux_dur, sampling_rate=sr, unit=unit)

    n_min = min(n_base, n_aux)
    n_max = max(n_base, n_aux)

    # init auxiliary buffer

    aux = AudioBuffer(aux_dur, sr, value=1.0, unit=unit)
    aux_012345 = AudioBuffer.from_array(list(range(5)), sr)

    # default mix

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux)(base)
        assert np.sum(np.abs(base._data[n_min:])) == 0
        np.testing.assert_equal(base._data[:n_min], aux._data[:n_min])

    # loop auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, loop_aux=True)(base)
        np.testing.assert_equal(base._data, np.ones(n_base))

    # extend base

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, extend_base=True)(base)
        assert len(base) == n_max
        np.testing.assert_equal(base._data[:n_aux], np.ones(n_aux))

    # restrict length of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, read_dur_aux=1, unit='samples')(base)
        assert base._data[0] == 1 and np.sum(np.abs(base._data[1:])) == 0

    # read position of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, read_pos_aux=n_aux - 1, unit='samples')(base)
        assert base._data[0] == 1 and np.sum(np.abs(base._data[1:])) == 0

    for i in range(5):
        with AudioBuffer(base_dur, sr, unit=unit) as base:
            Mix(aux_012345, read_pos_aux=i, unit='samples')(base)
            np.testing.assert_equal(base._data[:len(aux_012345) - i],
                                    aux_012345._data[i:])

    # write position of base

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, write_pos_base=n_base - 1, unit='samples')(base)
        assert base._data[-1] == 1 and np.sum(np.abs(base._data[:-1])) == 0

    # set gain of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, gain_aux_db=to_db(2), loop_aux=True)(base)
        assert all(base._data == 2)
        Mix(aux, gain_base_db=to_db(0.5), loop_aux=True)(base)
        assert all(base._data == 2)

    # clipping

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(aux, gain_aux_db=to_db(2), loop_aux=True, clip_mix=True)(base)
        assert all(base._data == 1)

    aux.free()
    aux_012345.free()


@pytest.mark.parametrize('base_dur,aux_dur,sr,unit',
                         [(1.0, 1.0, 8000, None),
                          (16000, 8000, 16000, 'samples'),
                          (500, 1000, 44100, 'ms')])
def test_append(base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = to_samples(base_dur, sampling_rate=sr, unit=unit)
    n_aux = to_samples(aux_dur, sampling_rate=sr, unit=unit)

    aux = AudioBuffer(aux_dur, sr, unit=unit, value=1.0)

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Append(aux)(base)
        np.testing.assert_equal(base._data[:n_base], np.zeros(n_base))
        np.testing.assert_equal(base._data[n_base:], np.ones(n_aux))

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Append(aux, read_pos_aux=n_aux - 1, unit='samples')(base)
        np.testing.assert_equal(base._data[:n_base], np.zeros(n_base))
        assert len(base._data) == n_base + 1
        assert base._data[-1] == 1

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Append(aux, read_dur_aux=1, unit='samples')(base)
        np.testing.assert_equal(base._data[:n_base], np.zeros(n_base))
        assert len(base._data) == n_base + 1
        assert base._data[-1] == 1

    aux.free()


@pytest.mark.parametrize(
    'start,dur,sr,unit',
    [
        (0.1, 0.8, 8000, 'seconds'),
        (0.25, 0.5, 8000, 'seconds'),
        (0.1, None, 8000, 'seconds'),
        (0.1, 0.8, 16000, 'seconds'),
        (0.25, 0.5, 16000, 'seconds'),
        (0.1, None, 16000, 'seconds'),
        pytest.param(
            -1.0, 0.8, 8000, 'seconds',
            marks=pytest.mark.xfail(raises=ValueError),
        ),  # negative start (seconds) -> error from pyauglib
        pytest.param(
            -1, 10, 8000, 'samples',
            marks=pytest.mark.xfail(raises=ValueError)
        ),  # negative start -> error from pyauglib
        pytest.param(
            0, -1, 8000, 'samples',
            marks=pytest.mark.xfail(raises=ValueError)
        ),  # negative duration -> error from pyauglib
        pytest.param(
            42.0, 0.5, 8000, 'seconds',
            marks=pytest.mark.xfail(raises=LibraryException)
        ),  # start > buffer length -> error from auglib (C++ side)
    ]
)
def test_trim(start, dur, sr, unit):
    # start from a step signal (0.5s of zeros, then 0.5s of ones) and trim out
    # different portions
    block_dur = 0.5
    target_start = start
    if target_start is None:
        target_start = 0
    target_dur = dur
    if target_dur is None:
        target_dur = 2 * block_dur - target_start
    target_length = int(round(target_dur * sr))

    try:
        target_array = np.append(
            np.zeros(int((block_dur - target_start) * sr)),
            np.ones(int((target_start + target_dur - block_dur) * sr))
        )
    except ValueError:
        target_array = None

    with AudioBuffer(block_dur, sr) as buf:
        AppendValue(block_dur, value=1.0)(buf)
        Trim(start_pos=start, duration=dur, unit=unit)(buf)
        assert len(buf) == target_length
        np.testing.assert_equal(target_array, buf._data)


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_normalize(n, sr):

    with AudioBuffer.from_array(np.linspace(-0.5, 0.5, num=n), sr) as buf:
        NormalizeByPeak()(buf)
        assert np.abs(buf._data).max() == 1.0


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_clip(n, sr):

    x = np.c_[np.random.uniform(-1.5, 1.0, n // 2),
              np.random.uniform(1.0, 1.5, n - (n // 2))]
    np.random.shuffle(x)

    with AudioBuffer.from_array(x, sr) as buf:
        Clip()(buf)
        np.isclose(np.abs(buf._data).max(), 1.0)

    with AudioBuffer.from_array(x, sr) as buf:
        Clip(threshold=0.5)(buf)
        np.isclose(np.abs(buf._data).max(), 0.5)

    with AudioBuffer.from_array(x, sr) as buf:
        Clip(threshold=0.5, normalize=True)(buf)
        assert np.isclose(np.abs(buf._data).max(), 1.0)

    x = np.random.uniform(1, 2, n)

    with AudioBuffer.from_array(x, sr) as buf:
        ClipByRatio(0.5, normalize=False)(buf)
        assert buf._data.max() <= np.median(x)

    with AudioBuffer.from_array(x, sr) as buf:
        ClipByRatio(0.5, normalize=True)(buf)
        assert np.isclose(np.abs(buf._data).max(), 1.0)
        assert np.abs(buf._data).min() >= 1. / 1.5


@pytest.mark.parametrize('n,sr,gain,max_peak,clip',
                         [(10, 8000, 20.0, None, False),
                          (10, 44100, 20.0, None, True),
                          (10, 44100, 20.0, 10.0, False),
                          (10, 44100, 20.0, 10.0, True)])
def test_gain_stage(n, sr, gain, max_peak, clip):

    x = np.random.uniform(-0.1, 1.0, n)
    with AudioBuffer.from_array(x, sr) as buf:
        GainStage(gain, max_peak_db=max_peak, clip=clip)(buf)
        if clip:
            assert np.abs(buf._data).max() <= 1.0
        elif max_peak is not None:
            assert np.isclose(np.abs(buf._data).max(), from_db(max_peak))
        else:
            assert np.isclose(np.abs(buf._data).max(),
                              from_db(gain) * np.abs(x).max())


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_fft_convolve(n, sr):

    with AudioBuffer(n, sr, unit='samples') as base:
        with AudioBuffer(n, sr, unit='samples', value=1.0) as aux:
            base._data[0] = 1
            FFTConvolve(aux, keep_tail=False)(base)
            np.testing.assert_equal(base._data, np.ones(len(base)))


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_filter(n, sr):
    # generate a boxcar signal (step up...step down)
    sig_in = np.zeros(n * sr, dtype='float32')
    sig_in[int(n * sr / 4):int(n * sr * 3 / 4)] = 1.0

    b, a = signal.butter(1, 0.5, 'lowpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        LowPass(0.5 * sr / 2)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    b, a = signal.butter(1, 0.5, 'highpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        HighPass(0.5 * sr / 2)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    b, a = signal.butter(1, np.array([0.5 - 0.25, 0.5 + 0.25]), 'bandpass')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        BandPass(0.5 * sr / 2, 0.5 * sr / 2)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    b, a = signal.butter(1, (2.0 / sr) * np.array([1000 - 5, 1000 + 5]),
                         'bandstop')
    sig_out = signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        BandStop(1000, 10)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)


@pytest.mark.parametrize(
    'filter_obj, params',
    [
        (LowPass, [1]),
        (HighPass, [1]),
        (BandPass, [1, 1]),
        (BandStop, [1, 1]),
    ]
)
def test_filter_errors(filter_obj, params):
    with pytest.raises(ValueError):
        design = 'non-supported'
        filter_obj(*params, design=design)


@pytest.mark.parametrize('dur,sr,gain,seed',
                         [(1.0, 8000, 0.0, 1)])
def test_WhiteNoiseUniform(dur, sr, gain, seed):

    auglib.seed(seed)
    with WhiteNoiseUniform(gain_db=gain)(AudioBuffer(dur, sr)) as noise:
        with AudioBuffer(len(noise), noise.sampling_rate,
                         unit='samples') as buf:
            auglib.seed(seed)
            lib.AudioBuffer_addWhiteNoiseUniform(buf._obj, gain)
            np.testing.assert_equal(noise._data, buf._data)
            np.testing.assert_almost_equal(buf._data.mean(), 0, decimal=1)


@pytest.mark.parametrize('dur,sr,gain,stddev,seed',
                         [(1.0, 8000, 0.0, 0.3, 1)])
def test_WhiteNoiseGaussian(dur, sr, gain, stddev, seed):

    auglib.seed(seed)
    with WhiteNoiseGaussian(stddev=stddev, gain_db=gain)(AudioBuffer(dur, sr))\
            as noise:
        with AudioBuffer(len(noise), noise.sampling_rate,
                         unit='samples') as buf:
            auglib.seed(seed)
            lib.AudioBuffer_addWhiteNoiseGaussian(buf._obj, gain, stddev)
            np.testing.assert_equal(noise._data, buf._data)
            np.testing.assert_almost_equal(buf._data.mean(), 0, decimal=1)
            np.testing.assert_almost_equal(buf._data.std(), stddev, decimal=1)


@pytest.mark.parametrize('dur,sr,gain,seed',
                         [(1.0, 8000, 0.0, 1)])
def test_PinkNoise(dur, sr, gain, seed):

    auglib.seed(seed)
    with PinkNoise(gain_db=gain)(AudioBuffer(dur, sr)) as noise:
        with AudioBuffer(len(noise), noise.sampling_rate,
                         unit='samples') as buf:
            auglib.seed(seed)
            lib.AudioBuffer_addPinkNoise(buf._obj, gain)
            np.testing.assert_equal(noise._data, buf._data)
            np.testing.assert_almost_equal(buf._data.mean(), 0, decimal=1)


@pytest.mark.parametrize('freq', [1, 440])
def test_sine(freq):

    sr = 8000
    n = sr

    with Tone(freq, shape='sine')(AudioBuffer(n, sr, unit='samples'))\
            as tone:
        sine = np.sin((np.arange(n, dtype=np.float) / sr) * 2 * np.pi * freq)
        np.testing.assert_almost_equal(tone._data, sine, decimal=3)


def test_tone_errors():
    with pytest.raises(ValueError):
        Tone(440, shape='non-supported')


def test_trim_errors():
    with pytest.raises(ValueError):
        Trim(fill='non-supported')


@pytest.mark.parametrize(
    'dur,sr',
    [
        (1.0, 8000,)
    ]
)
def test_compression(dur, sr):
    with Tone(220.0, shape='square')(AudioBuffer(dur, sr)) as tone:
        NormalizeByPeak(peak_db=-3.0)(tone)
        CompressDynamicRange(-12.0, 20.0, attack_time=0.0, release_time=0.1,
                             knee_radius_db=6.0, makeup_db=None)(tone)
        peak1 = tone.peak_db
    with Tone(220.0, shape='square')(AudioBuffer(dur, sr)) as tone:
        NormalizeByPeak(peak_db=-3.0)(tone)
        CompressDynamicRange(-12.0, 20.0, attack_time=0.0, release_time=0.1,
                             knee_radius_db=6.0, makeup_db=0.0)(tone)
        peak2 = tone.peak_db

    assert peak1 > peak2
    assert np.isclose(peak1, -3.0)


@pytest.mark.parametrize(
    'bit_rate', [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]
)
def test_AMRNB(bit_rate):

    original_wav = 'tests/test-assets/opensmile.wav'
    target_wav = f'tests/test-assets/opensmile_amrnb_rate{bit_rate}_ffmpeg.wav'

    transform = AMRNB(bit_rate)
    with AudioBuffer.read(original_wav) as buf:
        transform(buf)
        result = buf._data
        with AudioBuffer.read(target_wav) as target_buf:
            target = target_buf._data
            length = min(result.size, target.size)
            np.testing.assert_allclose(
                result[:length], target[:length], rtol=0.0, atol=0.065
            )


def test_function():

    def func_plus_c(x, sr, c):
        return x + c

    def func_halve(x, sr):
        return x[:, ::2]

    def func_times_2(x, sr):  # inplace
        x *= 2

    with AudioBuffer(20, 16000, unit='samples') as buffer:

        np.testing.assert_equal(
            buffer._data,
            np.zeros(20, dtype=np.float32),
        )

        # add 1 to buffer
        Function(func_plus_c, {'c': auglib.observe.IntUni(1, 1)})(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(20, dtype=np.float32),
        )

        # halve buffer size
        Function(func_halve)(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(10, dtype=np.float32),
        )

        # multiple by 2
        Function(func_times_2)(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(10, dtype=np.float32) * 2,
        )

        # double buffer size
        Function(lambda x, sr: np.tile(x, 2))(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(20, dtype=np.float32) * 2,
        )
