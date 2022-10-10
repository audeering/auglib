import os
import re

import audresample
import numpy as np
import pytest
import scipy

import auglib
from auglib import AudioBuffer
from auglib.core.buffer import lib
from auglib.transform import (
    AMRNB,
    Append,
    AppendValue,
    BabbleNoise,
    BandPass,
    BandStop,
    Base,
    Clip,
    ClipByRatio,
    CompressDynamicRange,
    FFTConvolve,
    Function,
    GainStage,
    HighPass,
    LowPass,
    Mask,
    Mix,
    NormalizeByPeak,
    Prepend,
    PrependValue,
    PinkNoise,
    Shift,
    Tone,
    Trim,
    WhiteNoiseUniform,
    WhiteNoiseGaussian,
)
from auglib.utils import (
    from_db,
    to_db,
    to_samples,
)


def rms_db(signal):
    return 20 * np.log10(np.sqrt(np.mean(np.square(signal))))


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'bypass_prob, preserve_level, base, expected',
    [
        (None, False, [1, 1], [2, 2]),
        (None, True, [1, 1], [1, 1]),
        (1, False, [0, 0], [0, 0]),
        (1, True, [0, 0], [0, 0]),
    ],
)
def test_Base(sampling_rate, bypass_prob, preserve_level, base, expected):

    # Define transform without aux
    class Transform(Base):

        def __init__(self, bypass_prob, preserve_level):
            super().__init__(
                bypass_prob=bypass_prob,
                preserve_level=preserve_level,
            )

        def _call(self, base):
            base._data = base._data + 1
            return base

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        t = Transform(bypass_prob, preserve_level)
        assert t.bypass_prob == bypass_prob
        assert t.preserve_level == preserve_level
        t(base_buf)
        np.testing.assert_almost_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
            decimal=4,
        )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[0, 0]])
@pytest.mark.parametrize('from_file', [True, False])
@pytest.mark.parametrize('observe', [True, False])
@pytest.mark.parametrize(
    'transform, preserve_level, aux, expected',
    [
        (None, False, [1, 1], [1, 1]),
        (None, True, [1, 1], [1.e-06, 1.e-06]),
        (Function(lambda x, sr: x + 1), False, [1, 1], [2, 2]),
        (Function(lambda x, sr: x + 1), True, [1, 1], [1.e-06, 1.e-06]),
    ],
)
def test_Base_aux(
        tmpdir,
        sampling_rate,
        base,
        from_file,
        observe,
        transform,
        preserve_level,
        aux,
        expected,
):

    # Define transform with aux
    class Transform(Base):

        def __init__(self, aux, preserve_level, transform):
            super().__init__(
                preserve_level=preserve_level,
                aux=aux,
                transform=transform,
            )

        def _call(self, base, aux):
            base._data = base._data + aux._data
            return base

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
            if from_file:
                path = os.path.join(tmpdir, 'test.wav')
                aux_buf.write(path)
                aux_buf.free()
                aux_buf = path
            if observe:
                aux_buf = auglib.observe.List([aux_buf])
            t = Transform(aux_buf, preserve_level, transform)
            assert t.bypass_prob is None
            assert t.preserve_level == preserve_level
            # unless buffer is read from file
            # we skip the following test
            # as we cannot serialize a buffer,
            # which is required to calculate its ID
            if from_file:
                assert t.aux == aux_buf
            assert t.transform == transform
            t(base_buf)
            np.testing.assert_almost_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
                decimal=4,
            )


# Test gain and SNR for BabbleNoise
@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('num_speakers', [1, 3])
@pytest.mark.parametrize(
    'gain_db, snr_db',
    [
        (0, None),
        (-10, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_BabbleNoise_1(
        duration,
        sampling_rate,
        num_speakers,
        gain_db,
        snr_db,
):
    auglib.seed(0)

    with AudioBuffer(
            duration,
            sampling_rate,
            value=1,
    ) as speech:

        transform = BabbleNoise(
            [speech],
            num_speakers=num_speakers,
            gain_db=gain_db,
            snr_db=snr_db,
        )
        with transform(AudioBuffer(duration, sampling_rate)) as noise:

            if snr_db is not None:
                gain_db = -120 - snr_db
            gain = from_db(gain_db)
            expected_noise = gain * np.ones(int(duration * sampling_rate))

            np.testing.assert_almost_equal(
                noise._data,
                expected_noise,
                decimal=4,
            )


# Test shorter speech signals for BabbleNoise
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    # NOTE: expected signal depends on seed
    'speech, num_speakers, duration, expected_noise',
    [
        ([[1, 0], ], 1, 3, [0, 1, 0]),
        ([[1, 0], ], 3, 3, [0, 2 / 3, 1 / 3]),
    ],
)
def test_BabbleNoise_2(
        sampling_rate,
        speech,
        num_speakers,
        duration,
        expected_noise,
):
    seed = 0
    auglib.seed(seed)

    speech_bufs = [AudioBuffer.from_array(s, sampling_rate) for s in speech]
    transform = BabbleNoise(speech_bufs, num_speakers=num_speakers)
    with AudioBuffer(duration, sampling_rate, unit='samples') as noise:
        transform(noise)
        np.testing.assert_almost_equal(
            noise._data,
            np.array(expected_noise, dtype=np.float32),
            decimal=4,
        )
    for buf in speech_bufs:
        buf.free()


@pytest.mark.parametrize(
    'base_dur,aux_dur,sr,unit',
    [
        (1.0, 1.0, 8000, None),
        (16000, 8000, 16000, 'samples'),
        (500, 1000, 44100, 'ms'),
    ],
)
def test_Mix_1(tmpdir, base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = to_samples(base_dur, sampling_rate=sr, unit=unit)
    n_aux = to_samples(aux_dur, sampling_rate=sr, unit=unit)

    n_min = min(n_base, n_aux)

    # init auxiliary buffer
    with AudioBuffer(aux_dur, sr, value=1.0, unit=unit) as aux:

        # default mix

        with AudioBuffer(base_dur, sr, unit=unit) as base:
            Mix(aux)(base)
            expected_mix = np.concatenate(
                [np.ones(n_min), np.zeros(n_base - n_min)]
            )
            np.testing.assert_equal(base._data, expected_mix)

        # clipping

        with AudioBuffer(base_dur, sr, unit=unit) as base:
            Mix(aux, gain_aux_db=to_db(2), loop_aux=True, clip_mix=True)(base)
            expected_mix = np.ones(n_base)
            np.testing.assert_equal(base._data, expected_mix)

    # Test for repeated execution.
    values = [0, 1, 2, 3, 4]
    expected_mix = np.zeros(n_base)
    for n in range(len(values)):
        expected_mix += np.concatenate(
            [values[n:], np.zeros(n_base - len(values[n:]))]
        )
    # Shift aux by increasing read_pos_aux
    with AudioBuffer(base_dur, sr, unit=unit) as base:
        with AudioBuffer.from_array(values, sr) as aux:
            Mix(
                aux,
                read_pos_aux=auglib.observe.List(values),
                unit='samples',
                num_repeat=len(values),
            )(base)
            np.testing.assert_equal(base._data, expected_mix)
    # Shift aux by observe list of buffers
    with AudioBuffer(base_dur, sr, unit=unit) as base:
        Mix(
            auglib.observe.List(
                [
                    AudioBuffer.from_array(values[n:], sr)
                    for n in range(len(values))
                ]
            ),
            num_repeat=len(values),
        )(base)
        np.testing.assert_equal(base._data, expected_mix)


# All duration are given in samples for this test
@pytest.mark.parametrize('base_duration', [5, 10])
@pytest.mark.parametrize('aux_duration', [5, 10])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('write_pos_base', [0, 1])
@pytest.mark.parametrize('extend_base', [False, True])
@pytest.mark.parametrize('read_pos_aux', [0, 1])
@pytest.mark.parametrize('read_dur_aux', [None, 3, 6])
@pytest.mark.parametrize('loop_aux', [False, True])
@pytest.mark.parametrize('gain_base_db', [0, 10])
@pytest.mark.parametrize(
    'gain_aux_db, snr_db',
    [
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_Mix_2(
        base_duration,
        aux_duration,
        sampling_rate,
        write_pos_base,
        extend_base,
        read_pos_aux,
        read_dur_aux,
        loop_aux,
        gain_base_db,
        gain_aux_db,
        snr_db,
):

    aux_values = np.array(range(aux_duration))
    base_value = 0.1

    # Skip tests for loop_aux=True and read_dur_aux not None
    # as this is broken at the moment, see
    # https://gitlab.audeering.com/tools/pyauglib/-/issues/47
    if (
            loop_aux
            and read_dur_aux is not None
    ):
        return

    # Skip test for read_dur_aux longer than len(aux)
    # and gain_base_db different from 0
    # as this is borken at the moment, see
    # https://gitlab.audeering.com/tools/pyauglib/-/issues/76
    if (
            gain_base_db != 0
            and read_dur_aux is not None
            and read_dur_aux > len(aux_values)
    ):
        return

    with AudioBuffer.from_array(aux_values, sampling_rate) as aux:
        with AudioBuffer(
                base_duration,
                sampling_rate,
                value=base_value,
                unit='samples',
        ) as base:

            # Number of samples read for mix from aux
            if (
                    read_dur_aux is None
                    or read_dur_aux == 0
            ):
                len_mix_aux = len(aux) - read_pos_aux
            else:
                len_mix_aux = read_dur_aux

            # Number of samples available for mix in base
            len_mix_base = len(base) - write_pos_base

            # If number of samples available for mix in base
            # is smaller than the number of samples read from aux
            # we pad zeros to base if extend_base is `True`.
            # Otherwise we trim the aux signal.
            gain_base = auglib.utils.from_db(gain_base_db)
            expected_mix = gain_base * base_value * np.ones(len(base))
            if len_mix_aux > len_mix_base:
                if extend_base:
                    expected_mix = np.concatenate(
                        [expected_mix, np.zeros(len_mix_aux - len_mix_base)],
                    )
                else:
                    len_mix_aux = len_mix_base

            # read_dur_aux is allowed to extend aux buffer,
            # in this case zeros are padded at the end.
            # Those zeros will NOT be included in the RMS_dB calculation
            len_mix_aux = min(
                len(aux) - read_pos_aux,
                len_mix_aux,
            )
            aux_values = aux_values[read_pos_aux:read_pos_aux + len_mix_aux]

            # As we use a fixed signal for `base`
            # the RMS_db value is independent of signal length
            rms_db_base = rms_db(gain_base * base_value)
            rms_db_aux = rms_db(aux_values)

            # Get gain factor for aux
            if gain_aux_db is None:
                gain_aux_db = 0.0
            if snr_db is not None:
                gain_aux_db = rms_db_base - snr_db - rms_db_aux
            gain_aux = auglib.utils.from_db(gain_aux_db)

            # Add aux values to expected mix
            mix_start = write_pos_base
            expected_mix[mix_start:mix_start + len_mix_aux] += (
                gain_aux * aux_values
            )
            # If aux should be looped,
            # we have to repeat adding aux to the mix
            mix_start += len_mix_aux
            if loop_aux:
                while mix_start < len(expected_mix):
                    mix_end = min(
                        mix_start + len_mix_aux,
                        len(expected_mix),
                    )
                    expected_mix[mix_start:mix_end] += (
                        gain_aux * aux_values[:mix_end - mix_start]
                    )
                    mix_start += len_mix_aux

            transform = Mix(
                aux,
                gain_base_db=gain_base_db,
                gain_aux_db=gain_aux_db,
                snr_db=snr_db,
                write_pos_base=write_pos_base,
                read_pos_aux=read_pos_aux,
                read_dur_aux=read_dur_aux,
                extend_base=extend_base,
                unit='samples',
                loop_aux=loop_aux,
            )
            transform(base)

            np.testing.assert_almost_equal(
                base._data,
                expected_mix,
                decimal=5,
            )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[1, 1]])
@pytest.mark.parametrize(
    'read_pos_aux, read_dur_aux, unit, aux, expected',
    [
        (0, 0, 'samples', [0, 2], [1, 1, 0, 2]),
        (0, None, 'samples', [0, 2], [1, 1, 0, 2]),
        (1, 0, 'samples', [0, 2], [1, 1, 2]),
        (0, 1, 'samples', [0, 2], [1, 1, 0]),
    ],
)
def test_Append(
        tmpdir,
        sampling_rate,
        base,
        read_pos_aux,
        read_dur_aux,
        unit,
        aux,
        expected,
):
    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
            auglib.transform.Append(
                aux_buf,
                read_pos_aux=read_pos_aux,
                read_dur_aux=read_dur_aux,
                unit=unit,
            )(base_buf)
            np.testing.assert_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
            )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[1, 1]])
@pytest.mark.parametrize(
    'duration, unit, value, expected',
    [
        (0, 'samples', 0, [1, 1]),
        (0, 'seconds', 0, [1, 1]),
        (1, 'samples', 2, [1, 1, 2]),
        (2, 'samples', 2, [1, 1, 2, 2]),
    ],
)
def test_AppendValue(
        sampling_rate,
        base,
        duration,
        unit,
        value,
        expected,
):
    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        AppendValue(
            duration,
            value,
            unit=unit,
        )(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
        )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[1, 1]])
@pytest.mark.parametrize(
    'read_pos_aux, read_dur_aux, unit, aux, expected',
    [
        (0, None, 'samples', [0, 2], [0, 2, 1, 1]),
        (0, 0, 'samples', [0, 2], [0, 2, 1, 1]),
        (1, None, 'samples', [0, 2], [2, 1, 1]),
        (0, 1, 'samples', [0, 2], [0, 1, 1]),
    ],
)
def test_Prepend(
        tmpdir,
        sampling_rate,
        base,
        read_pos_aux,
        read_dur_aux,
        unit,
        aux,
        expected,
):
    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
            Prepend(
                aux_buf,
                read_pos_aux=read_pos_aux,
                read_dur_aux=read_dur_aux,
                unit=unit,
            )(base_buf)
            np.testing.assert_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
            )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[1, 1]])
@pytest.mark.parametrize(
    'duration, unit, value, expected',
    [
        (0, 'samples', 0, [1, 1]),
        (0, 'seconds', 0, [1, 1]),
        (1, 'samples', 2, [2, 1, 1]),
        (2, 'samples', 2, [2, 2, 1, 1]),
    ],
)
def test_PrependValue(
        sampling_rate,
        base,
        duration,
        unit,
        value,
        expected,
):
    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        PrependValue(
            duration,
            value,
            unit=unit,
        )(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
        )


# Trim tests that should be independent of fill
@pytest.mark.parametrize('fill', ['none', 'zeros', 'loop'])
@pytest.mark.parametrize(
    'start_pos, duration, unit, signal, expected_signal',
    [
        (0, None, 'samples', [1, 2, 3], [1, 2, 3]),
        (0, 2, 'samples', [1, 2, 3], [1, 2]),
        (1, 2, 'samples', [1, 2, 3], [2, 3]),
        # Errors raised by pyauglib
        pytest.param(  # negative start (seconds)
            -1.0, None, 'seconds', [1, 2, 3], None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # negative start
            -1, None, 'samples', [1, 2, 3], None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # negative duration
            0, -1, 'samples', [1, 2, 3], None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Errors raised by auglib (C++ library)
        pytest.param(  # start > buffer length
            4, None, 'samples', [1, 2, 3], None,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ]
)
def test_Trim1(fill, start_pos, duration, unit, signal, expected_signal):
    with AudioBuffer.from_array(signal, 8000) as buf:
        transform = Trim(
            start_pos=start_pos,
            duration=duration,
            fill=fill,
            unit=unit,
        )
        transform(buf)
        np.testing.assert_equal(buf._data, np.array(expected_signal))


# Trim tests that should dependent of fill
@pytest.mark.parametrize(
    'start_pos, duration, unit, fill, signal, expected_signal',
    [
        (2, 2, 'samples', 'none', [1, 2, 3], [3]),
        (2, 2, 'samples', 'zeros', [1, 2, 3], [3, 0]),
        (2, 2, 'samples', 'loop', [1, 2, 3], [3, 1]),
        (2, 3, 'samples', 'none', [1, 2, 3], [3]),
        (2, 3, 'samples', 'zeros', [1, 2, 3], [3, 0, 0]),
        (2, 3, 'samples', 'loop', [1, 2, 3], [3, 1, 2]),
        (2, 4, 'samples', 'none', [1, 2, 3], [3]),
        (2, 4, 'samples', 'zeros', [1, 2, 3], [3, 0, 0, 0]),
        (2, 4, 'samples', 'loop', [1, 2, 3], [3, 1, 2, 3]),
    ]
)
def test_Trim2(start_pos, duration, unit, fill, signal, expected_signal):
    with AudioBuffer.from_array(signal, 8000) as buf:
        transform = Trim(
            start_pos=start_pos,
            duration=duration,
            fill=fill,
            unit=unit,
        )
        transform(buf)
        np.testing.assert_equal(buf._data, np.array(expected_signal))


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_normalize(n, sr):

    with AudioBuffer.from_array(np.linspace(-0.5, 0.5, num=n), sr) as buf:
        NormalizeByPeak()(buf)
        assert np.abs(buf._data).max() == 1.0


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'threshold, normalize, signal, expected_signal',
    [
        (1.0, False, [-1.5, -0.5, 0.5, 1.5], [-1.0, -0.5, 0.5, 1.0]),
        (0.5, False, [-1.5, 0.5], [-0.5, 0.5]),
        (0.5, True, [-1.5, 0.5], [-1.0, 1.0]),
        (0.5, True, [-1.5, 0.25], [-1.0, 0.5]),
    ],
)
def test_Clip(sampling_rate, threshold, normalize, signal, expected_signal):

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform = Clip(
            threshold=auglib.utils.to_db(threshold),
            normalize=normalize,
        )
        transform(buf)
        np.testing.assert_almost_equal(
            buf._data,
            np.array(expected_signal, dtype=np.float32),
            decimal=4,
        )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'ratio, normalize, soft, signal, expected_signal',
    [
        (0.5, False, False, [0.5, 2.0], [0.5, 0.5]),
        (0.5, False, True, [0.5, 2.0], [0.5, 0.5]),
        (0.5, True, False, [0.5, 2.0], [1.0, 1.0]),
        (1 / 3, False, False, [0.5, 1.0, 2.0], [0.5, 1.0, 1.0]),
    ],
)
def test_ClipByRatio(
        sampling_rate,
        ratio,
        normalize,
        soft,
        signal,
        expected_signal,
):

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform = ClipByRatio(ratio, normalize=normalize, soft=soft)
        transform(buf)
        np.testing.assert_almost_equal(
            buf._data,
            np.array(expected_signal, dtype=np.float32),
            decimal=4,
        )


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


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'keep_tail, aux, base, expected',
    [
        (False, [1, 0, 0], [1, 2, 3], [1, 2, 3]),
        (False, [0, 1, 0], [1, 2, 3], [0, 1, 2]),
        (True, [0, 1, 0], [1, 2, 3], [0, 1, 2, 3, 0]),
        (True, [0, 1, 0, 0], [1, 2, 3], [0, 1, 2, 3, 0, 0]),
    ],
)
def test_FFTConvolve(sampling_rate, keep_tail, aux, base, expected):

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
            FFTConvolve(aux_buf, keep_tail=keep_tail)(base_buf)
            np.testing.assert_almost_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
                decimal=4,
            )


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_filter(n, sr):
    # generate a boxcar signal (step up...step down)
    sig_in = np.zeros(n * sr, dtype='float32')
    sig_in[int(n * sr / 4):int(n * sr * 3 / 4)] = 1.0

    b, a = scipy.signal.butter(1, 0.5, 'lowpass')
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        LowPass(0.5 * sr / 2)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    b, a = scipy.signal.butter(1, 0.5, 'highpass')
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        HighPass(0.5 * sr / 2)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    b, a = scipy.signal.butter(
        1,
        np.array([0.5 - 0.25, 0.5 + 0.25]),
        'bandpass',
    )
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        BandPass(0.5 * sr / 2, 0.5 * sr / 2)(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    b, a = scipy.signal.butter(
        1,
        (2.0 / sr) * np.array([1000 - 5, 1000 + 5]),
        'bandstop',
    )
    sig_out = scipy.signal.lfilter(b, a, sig_in)

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


@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'gain_db, snr_db',
    [
        (-10, None),
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_WhiteNoiseUniform(duration, sampling_rate, gain_db, snr_db):

    seed = 0
    auglib.seed(seed)
    transform = WhiteNoiseUniform(
        gain_db=gain_db,
        snr_db=snr_db,
    )
    with transform(AudioBuffer(duration, sampling_rate)) as noise:
        with AudioBuffer(
                len(noise),
                noise.sampling_rate,
                unit='samples',
        ) as expected_noise:
            if gain_db is None:
                gain_db = 0.0

            # Approximate value of the uniform white noise energy
            noise_volume = -4.77125
            if snr_db is not None:
                gain_db = -120 - snr_db - noise_volume

            expected_volume = noise_volume + gain_db
            expected_mean = 0

            # Create expected noise signal
            auglib.seed(seed)
            lib.AudioBuffer_addWhiteNoiseUniform(
                expected_noise._obj,
                gain_db,
            )
            # Double check volume is correct
            np.testing.assert_almost_equal(
                rms_db(expected_noise._data),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_equal(
                noise._data,
                expected_noise._data,
            )
            np.testing.assert_almost_equal(
                rms_db(noise._data),
                expected_volume,
                decimal=1,
            )
            np.testing.assert_almost_equal(
                noise._data.mean(),
                expected_mean,
                decimal=1,
            )


@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('stddev', [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize(
    'gain_db, snr_db',
    [
        (-10, None),
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_WhiteNoiseGaussian(duration, sampling_rate, stddev, gain_db, snr_db):

    seed = 0
    auglib.seed(seed)
    transform = WhiteNoiseGaussian(
        stddev=stddev,
        gain_db=gain_db,
        snr_db=snr_db,
    )
    with transform(AudioBuffer(duration, sampling_rate)) as noise:
        with AudioBuffer(
                len(noise),
                noise.sampling_rate,
                unit='samples',
        ) as expected_noise:

            if gain_db is None:
                gain_db = 0.0

            # Power of white noise is given by std^2
            noise_volume = 10 * np.log10(stddev ** 2)

            if snr_db is not None:
                # We add noise to an empty signal,
                # which is limited to -120 dB
                gain_db = -120 - snr_db - noise_volume

            expected_volume = noise_volume + gain_db
            expected_mean = 0
            expected_stddev = auglib.utils.from_db(gain_db) * stddev

            # Create expected noise signal
            auglib.seed(seed)
            lib.AudioBuffer_addWhiteNoiseGaussian(
                expected_noise._obj,
                gain_db,
                stddev,
            )
            # Double check volume is correct
            np.testing.assert_almost_equal(
                rms_db(expected_noise._data),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_equal(
                noise._data,
                expected_noise._data,
            )
            np.testing.assert_almost_equal(
                rms_db(noise._data),
                expected_volume,
                decimal=1,
            )
            np.testing.assert_almost_equal(
                noise._data.mean(),
                expected_mean,
                decimal=1,
            )
            np.testing.assert_almost_equal(
                noise._data.std(),
                expected_stddev,
                decimal=1,
            )


@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'gain_db, snr_db',
    [
        (-10, None),
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_PinkNoise(duration, sampling_rate, gain_db, snr_db):
    seed = 0
    auglib.seed(seed)
    transform = PinkNoise(
        gain_db=gain_db,
        snr_db=snr_db,
    )

    with transform(AudioBuffer(duration, sampling_rate)) as noise:
        with AudioBuffer(
                len(noise),
                noise.sampling_rate,
                unit='samples',
        ) as expected_noise:

            if gain_db is None:
                gain_db = 0.0

            # Get the empiric measure of the RMS energy for Pink Noise
            with AudioBuffer(
                    len(noise), noise.sampling_rate, unit='samples'
            ) as tmp_pink_noise:
                auglib.seed(seed)
                lib.AudioBuffer_addPinkNoise(
                    tmp_pink_noise._obj,
                    0,
                )
                noise_volume = rms_db(tmp_pink_noise._data)

                if snr_db is not None:
                    gain_db = -120 - snr_db - noise_volume

                expected_volume = noise_volume + gain_db

                if snr_db is not None:
                    lib.AudioBuffer_mix(
                        expected_noise._obj,
                        tmp_pink_noise._obj,
                        0,
                        gain_db,
                        0,
                        0,
                        0,
                        False,
                        False,
                        False,
                    )
                else:
                    auglib.seed(seed)
                    lib.AudioBuffer_addPinkNoise(
                        expected_noise._obj,
                        gain_db,
                    )

            # Check volume is correct
            np.testing.assert_almost_equal(
                rms_db(expected_noise._data),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_almost_equal(
                rms_db(noise._data),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_equal(
                noise._data,
                expected_noise._data,
            )


@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [96000])
@pytest.mark.parametrize('frequency', [1000])
@pytest.mark.parametrize('shape', ['sine', 'square', 'triangle', 'sawtooth'])
@pytest.mark.parametrize(
    'gain_db, snr_db',
    [
        (-10, None),
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_Tone(duration, sampling_rate, frequency, shape, gain_db, snr_db):

    transform = Tone(
        frequency,
        shape=shape,
        gain_db=gain_db,
        snr_db=snr_db,
    )
    with transform(AudioBuffer(duration, sampling_rate)) as tone:

        if gain_db is None:
            gain_db = 0.0

        # Expected root mean square values,
        # see https://en.wikipedia.org/wiki/Root_mean_square
        time = (
            np.arange(duration * sampling_rate, dtype=float)
            / sampling_rate
        )
        omega = 2 * np.pi * frequency
        if shape == 'sine':
            expected_tone = np.sin(omega * time)
            expected_rms_db = 20 * np.log10(1 / np.sqrt(2))
        elif shape == 'square':
            expected_tone = -1 * scipy.signal.square(omega * time, duty=0.5)
            expected_rms_db = 20 * np.log10(1)
        elif shape == 'triangle':
            expected_tone = -1 * scipy.signal.sawtooth(omega * time, width=0.5)
            expected_rms_db = 20 * np.log10(1 / np.sqrt(3))
        elif shape == 'sawtooth':
            expected_tone = scipy.signal.sawtooth(omega * time, width=1)
            expected_rms_db = 20 * np.log10(1 / np.sqrt(3))
        # Check reference signal has expected RMS value
        np.testing.assert_almost_equal(
            rms_db(expected_tone),
            expected_rms_db,
            decimal=2,
        )

        if snr_db is not None:
            # We add noise to an empty signal,
            # which is limited to -120 dB
            gain_db = -120 - snr_db - rms_db(expected_tone)

        expected_volume = rms_db(expected_tone) + gain_db

        # Add gain to expected tone
        gain = 10 ** (gain_db / 20)
        expected_tone *= gain

        # We cannot use np.testing.assert_almost_equal()
        # for comparing square, triangle, and sawtooth signals
        # as there might be a shift of one sample
        # from time to time,
        # which would let the test fail.
        assert np.mean(np.abs(tone._data - expected_tone)) < 1e4
        np.testing.assert_almost_equal(
            rms_db(tone._data),
            expected_volume,
            decimal=2,
        )


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


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'base, function, error, error_msg',
    [
        (
            [0, 0],
            lambda sig, sr: np.array([]),
            RuntimeError,
            (
                'Buffers must be non-empty. '
                'Yours is empty '
                'after applying the following transform: '
                "'$auglib.core.transform.Function"
            ),
        ),
    ],
)
def test_Function_errors(
        sampling_rate,
        base,
        function,
        error,
        error_msg,
):

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        with pytest.raises(error, match=re.escape(error_msg)):
            Function(function)(base_buf)


@pytest.mark.parametrize(
    'signal, sampling_rate, transform',
    [
        (
            np.zeros((1, 10)),
            8000,
            Function(lambda x, sr: x + 1),
        )
    ]
)
@pytest.mark.parametrize(
    'start_pos, duration, step, invert, expected',
    [
        (0, None, None, False, np.zeros((1, 10))),
        (0, None, None, True, np.ones((1, 10))),
        (0, 5, None, False, [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]),
        (0, 5, None, True, [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
        (3, 5, None, False, [[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]]),
        (3, 5, None, True, [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        (3, None, None, False, [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]),
        (3, None, None, True, [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
        (0, None, 2, False, [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]),
        (0, None, 2, True, [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]),
        (3, 5, 2, False, [[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]]),
        (3, 5, 2, True, [[0, 0, 0, 1, 1, 0, 0, 1, 0, 0]]),
        (0, None, (1, 3), False, [[0, 1, 1, 1, 0, 1, 1, 1, 0, 1]]),
        (0, None, (1, 3), True, [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0]]),
        (3, 5, (1, 3), False, [[1, 1, 1, 0, 1, 1, 1, 0, 1, 1]]),
        (3, 5, (1, 3), True, [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]),
    ]
)
def test_mask(signal, sampling_rate, transform, start_pos,
              duration, step, invert, expected):

    mask = Mask(
        transform,
        start_pos=start_pos,
        duration=duration,
        step=step,
        invert=invert,
        unit='samples',
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        mask(buf)
        augmented_signal = buf.to_array()
        np.testing.assert_equal(augmented_signal, expected)


@pytest.mark.parametrize(
    'signal, original_rate, target_rate',
    [
        (
            np.random.uniform(-1, 1, (1, 16000)).astype(np.float32),
            16000,
            8000,
        ),
        (
            np.random.uniform(-1, 1, (1, 16000)).astype(np.float32),
            16000,
            16000,
        ),
        (
            np.random.uniform(-1, 1, (1, 8000)).astype(np.float32),
            8000,
            16000,
        ),
    ]
)
@pytest.mark.parametrize(
    'override',
    [False, True],
)
def test_resample(signal, original_rate, target_rate, override):

    expected = audresample.resample(signal, original_rate, target_rate)

    transform = auglib.transform.Resample(target_rate, override=override)
    with AudioBuffer.from_array(signal, original_rate) as buf:
        transform(buf)
        resampled = buf.to_array()
        if override:
            assert buf.sampling_rate == target_rate
        else:
            assert buf.sampling_rate == original_rate

    np.testing.assert_equal(resampled, expected)


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'duration, unit, base, expected',
    [
        (0, 'samples', [1, 2, 3], [1, 2, 3]),
        (1, 'samples', [1, 2, 3], [2, 3, 1]),
        (2, 'samples', [1, 2, 3], [3, 1, 2]),
        (3, 'samples', [1, 2, 3], [1, 2, 3]),
        (4, 'samples', [1, 2, 3], [2, 3, 1]),
    ],
)
def test_Shift(sampling_rate, duration, unit, base, expected):
    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        Shift(duration=duration, unit=unit)(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
        )
