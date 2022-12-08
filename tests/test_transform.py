import os
import re

import audresample
import numpy as np
import pytest
import scipy

import audobject
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


# Define transform with aux
class TransformAux(Base):

    def __init__(self, aux, *, preserve_level=False, transform=None):
        super().__init__(
            preserve_level=preserve_level,
            aux=aux,
            transform=transform,
        )

    def _call(self, base, aux):
        base._data = base._data + aux._data
        return base


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

    transform = Transform(bypass_prob, preserve_level)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        assert transform.bypass_prob == bypass_prob
        assert transform.preserve_level == preserve_level
        transform(base_buf)
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

    with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
        if from_file:
            path = os.path.join(tmpdir, 'test.wav')
            aux_buf.write(path)
            aux_buf.free()
            aux_buf = path
        if observe:
            aux_buf = auglib.observe.List([aux_buf])
        base_transform = TransformAux(
            aux_buf,
            preserve_level=preserve_level,
            transform=transform,
        )
        if observe and not from_file:
            error_msg = (
                "Cannot serialize list if it contains an instance "
                "of <class 'auglib.core.buffer.AudioBuffer'>. "
                "As a workaround, save buffer to disk and add filename."
            )
            with pytest.raises(ValueError, match=error_msg):
                base_transform.to_yaml_s(include_version=False)

        elif isinstance(aux_buf, auglib.AudioBuffer):
            error_msg = (
                "Cannot serialize an instance "
                "of <class 'auglib.core.buffer.AudioBuffer'>. "
                "As a workaround, save buffer to disk and pass filename."
            )
            with pytest.raises(ValueError, match=error_msg):
                base_transform.to_yaml_s(include_version=False)
        else:
            base_transform = audobject.from_yaml_s(
                base_transform.to_yaml_s(include_version=False),
            )
        assert base_transform.bypass_prob is None
        assert base_transform.preserve_level == preserve_level
        # unless buffer is read from file
        # we skip the following test
        # as we cannot serialize a buffer,
        # which is required to calculate its ID
        if from_file:
            assert base_transform.aux == aux_buf
        assert base_transform.transform == transform
        with AudioBuffer.from_array(base, sampling_rate) as base_buf:
            base_transform(base_buf)
            np.testing.assert_almost_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
                decimal=4,
            )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[0, 0]])
@pytest.mark.parametrize(
    'aux, transform, expected',
    [
        (
            Function(lambda x, sr: x + 1),
            None,
            [1, 1],
        ),
        (
            Function(lambda x, sr: x + 1),
            Function(lambda x, sr: x + 1),
            [2, 2],
        ),
    ]
)
def test_Base_aux_transform(sampling_rate, base, aux, transform, expected):

    transform = TransformAux(aux, transform=transform)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_almost_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
            decimal=4,
        )


@pytest.mark.parametrize(
    'bit_rate', [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]
)
def test_AMRNB(bit_rate):

    original_wav = 'tests/test-assets/opensmile.wav'
    target_wav = f'tests/test-assets/opensmile_amrnb_rate{bit_rate}_ffmpeg.wav'

    transform = AMRNB(bit_rate)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.read(original_wav) as buf:
        transform(buf)
        result = buf._data
        with AudioBuffer.read(target_wav) as target_buf:
            target = target_buf._data
            length = min(result.size, target.size)
            np.testing.assert_allclose(
                result[:length], target[:length], rtol=0.0, atol=0.065
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
    with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:

        transform = auglib.transform.Append(
            aux_buf,
            read_pos_aux=read_pos_aux,
            read_dur_aux=read_dur_aux,
            unit=unit,
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with AudioBuffer.from_array(base, sampling_rate) as base_buf:
            transform(base_buf)
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
    transform = AppendValue(
        duration,
        value,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False)
    )

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
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
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
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
        ([[1, 0], ], 1, 4, [1, 0, 1, 0]),
        ([[1, 0, 0], ], 1, 3, [0, 0, 1]),
        ([[0, 1], ], 3, 3, [2 / 3, 1 / 3, 2 / 3]),
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
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer(duration, sampling_rate, unit='samples') as noise:
        transform(noise)
        np.testing.assert_almost_equal(
            noise._data,
            np.array(expected_noise, dtype=np.float32),
            decimal=4,
        )
    for buf in speech_bufs:
        buf.free()


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

    transform = Clip(
        threshold=auglib.utils.to_db(threshold),
        normalize=normalize,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
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

    transform = ClipByRatio(ratio, normalize=normalize, soft=soft)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        np.testing.assert_almost_equal(
            buf._data,
            np.array(expected_signal, dtype=np.float32),
            decimal=4,
        )


@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [8000])
def test_Compression(duration, sampling_rate):
    transform = CompressDynamicRange(
        -12.0,
        20.0,
        attack_time=0.0,
        release_time=0.1,
        knee_radius_db=6.0,
        makeup_db=None,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer(duration, sampling_rate) as buf:
        Tone(220.0, shape='square')(buf)
        NormalizeByPeak(peak_db=-3.0)(buf)
        transform(buf)
        peak1 = buf.peak_db

    transform = CompressDynamicRange(
        -12.0,
        20.0,
        attack_time=0.0,
        release_time=0.1,
        knee_radius_db=6.0,
        makeup_db=0.0,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer(duration, sampling_rate) as buf:
        Tone(220.0, shape='square')(buf)
        NormalizeByPeak(peak_db=-3.0)(buf)
        transform(buf)
        peak2 = buf.peak_db

    assert peak1 > peak2
    assert np.isclose(peak1, -3.0)


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

    with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
        transform = FFTConvolve(aux_buf, keep_tail=keep_tail)
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with AudioBuffer.from_array(base, sampling_rate) as base_buf:
            transform(base_buf)
            np.testing.assert_almost_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
                decimal=4,
            )


# BandPass, BandStop, HighPass, LowPass
@pytest.mark.parametrize('sr', [8000, 44100])
@pytest.mark.parametrize('n', [10])
def test_filter(n, sr):
    # generate a boxcar signal (step up...step down)
    sig_in = np.zeros(n * sr, dtype='float32')
    sig_in[int(n * sr / 4):int(n * sr * 3 / 4)] = 1.0

    # Lowpass
    b, a = scipy.signal.butter(1, 0.5, 'lowpass')
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    transform = LowPass(0.5 * sr / 2)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        transform(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    # Highpass
    b, a = scipy.signal.butter(1, 0.5, 'highpass')
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    transform = HighPass(0.5 * sr / 2)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        transform(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    # Bandpass
    b, a = scipy.signal.butter(
        1,
        np.array([0.5 - 0.25, 0.5 + 0.25]),
        'bandpass',
    )
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    transform = BandPass(0.5 * sr / 2, 0.5 * sr / 2)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        transform(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)

    # Bandstop
    b, a = scipy.signal.butter(
        1,
        (2.0 / sr) * np.array([1000 - 5, 1000 + 5]),
        'bandstop',
    )
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    transform = BandStop(1000, 10)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        transform(buf)
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


def test_Function():

    def func_double(x, sr):
        import numpy as np
        return np.tile(x, 2)

    def func_plus_c(x, sr, c):
        return x + c

    def func_times_2(x, sr):  # inplace
        x *= 2

    with AudioBuffer(20, 16000, unit='samples') as buffer:

        np.testing.assert_equal(
            buffer._data,
            np.zeros(20, dtype=np.float32),
        )

        # add 1 to buffer
        transform = Function(func_plus_c, {'c': auglib.observe.IntUni(1, 1)})
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(20, dtype=np.float32),
        )

        # halve buffer size
        transform = Function(lambda x, sr: x[:, ::2])
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(10, dtype=np.float32),
        )

        # multiple by 2
        transform = Function(func_times_2)
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(10, dtype=np.float32) * 2,
        )

        # double buffer size
        transform = Function(func_double)
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
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


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('signal', [np.random.uniform(-0.1, 1.0, 10)])
@pytest.mark.parametrize('gain', [20.0])
@pytest.mark.parametrize('max_peak', [None, 10.0])
@pytest.mark.parametrize('clip', [False, True])
def test_GainStage(sampling_rate, signal, gain, max_peak, clip):

    transform = GainStage(gain, max_peak_db=max_peak, clip=clip)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        if clip:
            assert np.abs(buf._data).max() <= 1.0
        elif max_peak is not None:
            assert np.isclose(np.abs(buf._data).max(), from_db(max_peak))
        else:
            assert np.isclose(np.abs(buf._data).max(),
                              from_db(gain) * np.abs(signal).max())


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
        (0, 0, None, False, np.ones((1, 10))),
        (0, 0, None, True, np.zeros((1, 10))),
        (0, 5, None, False, [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]),
        (0, 5, None, True, [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
        (3, 5, None, False, [[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]]),
        (3, 5, None, True, [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        (3, None, None, False, [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]),
        (3, None, None, True, [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
        (3, 0, None, False, np.ones((1, 10))),
        (3, 0, None, True, np.zeros((1, 10))),
        (0, None, 2, False, [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]),
        (0, None, 2, True, [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]),
        (0, 0, 2, False, np.ones((1, 10))),
        (0, 0, 2, True, np.zeros((1, 10))),
        (3, 5, 2, False, [[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]]),
        (3, 5, 2, True, [[0, 0, 0, 1, 1, 0, 0, 1, 0, 0]]),
        (0, None, (1, 3), False, [[0, 1, 1, 1, 0, 1, 1, 1, 0, 1]]),
        (0, None, (1, 3), True, [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0]]),
        (0, 0, (1, 3), False, np.ones((1, 10))),
        (0, 0, (1, 3), True, np.zeros((1, 10))),
        (3, 5, (1, 3), False, [[1, 1, 1, 0, 1, 1, 1, 0, 1, 1]]),
        (3, 5, (1, 3), True, [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]),
    ]
)
def test_Mask(signal, sampling_rate, transform, start_pos,
              duration, step, invert, expected):

    mask = Mask(
        transform,
        start_pos=start_pos,
        duration=duration,
        step=step,
        invert=invert,
        unit='samples',
    )
    mask.to_yaml_s(include_version=False)

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        mask(buf)
        augmented_signal = buf.to_array()
        np.testing.assert_equal(augmented_signal, expected)


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
        transform = Mix(aux)
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with AudioBuffer(base_dur, sr, unit=unit) as base:
            transform(base)
            expected_mix = np.concatenate(
                [np.ones(n_min), np.zeros(n_base - n_min)]
            )
            np.testing.assert_equal(base._data, expected_mix)

        # clipping
        transform = Mix(
            aux,
            gain_aux_db=to_db(2),
            loop_aux=True,
            clip_mix=True,
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with AudioBuffer(base_dur, sr, unit=unit) as base:
            transform(base)
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
    with AudioBuffer.from_array(values, sr) as aux:
        transform = Mix(
            aux,
            read_pos_aux=auglib.observe.List(values),
            unit='samples',
            num_repeat=len(values),
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with AudioBuffer(base_dur, sr, unit=unit) as base:
            transform(base)
            np.testing.assert_equal(base._data, expected_mix)

    # Shift aux by observe list of buffers
    transform = Mix(
        auglib.observe.List(
            [
                AudioBuffer.from_array(values[n:], sr)
                for n in range(len(values))
            ]
        ),
        num_repeat=len(values),
    )
    error_msg = (
        "Cannot serialize list if it contains an instance "
        "of <class 'auglib.core.buffer.AudioBuffer'>. "
        "As a workaround, save buffer to disk and add filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        transform(base)
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


@pytest.mark.parametrize('sampling_rate', [8000, 44100])
@pytest.mark.parametrize('signal', [np.linspace(-0.5, 0.5, num=10)])
def test_Normalize(sampling_rate, signal):

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform = NormalizeByPeak()
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buf)
        assert np.abs(buf._data).max() == 1.0


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
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
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
    with AudioBuffer.from_array(aux, sampling_rate) as aux_buf:

        transform = Prepend(
            aux_buf,
            read_pos_aux=read_pos_aux,
            read_dur_aux=read_dur_aux,
            unit=unit,
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with AudioBuffer.from_array(base, sampling_rate) as base_buf:
            transform(base_buf)
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
    transform = PrependValue(
        duration,
        value,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
        )


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
def test_Resample(signal, original_rate, target_rate, override):

    expected = audresample.resample(signal, original_rate, target_rate)

    transform = auglib.transform.Resample(target_rate, override=override)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

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
        (None, 'samples', [1, 2, 3], [1, 2, 3]),
        (0, 'samples', [1, 2, 3], [1, 2, 3]),
        (1, 'samples', [1, 2, 3], [2, 3, 1]),
        (2, 'samples', [1, 2, 3], [3, 1, 2]),
        (3, 'samples', [1, 2, 3], [1, 2, 3]),
        (4, 'samples', [1, 2, 3], [2, 3, 1]),
    ],
)
def test_Shift(sampling_rate, duration, unit, base, expected):

    transform = Shift(duration=duration, unit=unit)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
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
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
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


def test_Tone_errors():
    with pytest.raises(ValueError):
        Tone(440, shape='non-supported')


# Trim tests that should be independent of fill
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('fill', ['none', 'zeros', 'loop'])
@pytest.mark.parametrize(
    'start_pos, end_pos, duration, unit, signal, expected_signal',
    [
        (0, None, None, 'samples', [1, 2, 3], [1, 2, 3]),
        (0, None, 0, 'samples', [1, 2, 3], [1, 2, 3]),
        (0, None, 0, 'seconds', [1, 2, 3], [1, 2, 3]),
        (0, None, 2, 'samples', [1, 2, 3], [1, 2]),
        (1, None, 2, 'samples', [1, 2, 3], [2, 3]),
        (None, 1, 1, 'samples', [1, 2, 3], [2]),
    ]
)
def test_Trim(
        sampling_rate,
        fill,
        start_pos,
        end_pos,
        duration,
        unit,
        signal,
        expected_signal,
):
    transform = Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        np.testing.assert_equal(buf._data, np.array(expected_signal))


# Trim, fill='none'
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('unit', ['samples'])
@pytest.mark.parametrize('signal', [[1, 2, 3, 4]])
@pytest.mark.parametrize('fill', ['none'])
@pytest.mark.parametrize('fill_pos', ['right', 'left', 'both'])
@pytest.mark.parametrize(
    'start_pos, end_pos, duration, expected_signal',
    [
        (None, None, None, [1, 2, 3, 4]),
        (None, None, 0, [1, 2, 3, 4]),
        (None, None, 2, [2, 3]),
        (None, None, 3, [1, 2, 3]),
        (None, None, 6, [1, 2, 3, 4]),
        (None, 2, None, [1, 2]),
        (None, 2, 0, [1, 2]),
        (None, 2, 3, [1, 2]),
        (0, None, None, [1, 2, 3, 4]),
        (0, None, 0, [1, 2, 3, 4]),
        (2, None, 3, [3, 4]),
        (0, 0, None, [1, 2, 3, 4]),
        (0, 2, None, [1, 2]),
        (2, 0, None, [3, 4]),
        (0, 0, 0, [1, 2, 3, 4]),
        (0, 2, 0, [1, 2]),
        (2, 0, 0, [3, 4]),
        (0, 2, 3, [1, 2]),
        (0, 2, 4, [1, 2]),
        (2, 0, 3, [3, 4]),
        (2, 0, 4, [3, 4]),
        (1, 1, 1, [2]),
        (1, 1, 3, [2, 3]),
    ]
)
def test_Trim_fill_none(
        sampling_rate,
        unit,
        signal,
        fill,
        fill_pos,
        start_pos,
        end_pos,
        duration,
        expected_signal,
):
    transform = Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        fill_pos=fill_pos,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        np.testing.assert_equal(buf._data, np.array(expected_signal))


# Trim, fill='zeros'
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('unit', ['samples'])
@pytest.mark.parametrize('signal', [[1, 2, 3, 4]])
@pytest.mark.parametrize('fill', ['zeros'])
@pytest.mark.parametrize(
    'start_pos, end_pos, duration, fill_pos, expected_signal',
    [
        (None, None, 2, 'right', [2, 3]),
        (None, None, 3, 'right', [1, 2, 3]),
        (None, None, 6, 'right', [1, 2, 3, 4, 0, 0]),
        (None, None, 6, 'left', [0, 0, 1, 2, 3, 4]),
        (None, None, 6, 'both', [0, 1, 2, 3, 4, 0]),
        (None, 2, 3, 'right', [1, 2, 0]),
        (None, 2, 3, 'left', [0, 1, 2]),
        (None, 2, 3, 'both', [1, 2, 0]),
        (2, None, 3, 'right', [3, 4, 0]),
        (2, None, 3, 'left', [0, 3, 4]),
        (2, None, 3, 'both', [3, 4, 0]),
        (2, None, 4, 'right', [3, 4, 0, 0]),
        (2, None, 4, 'left', [0, 0, 3, 4]),
        (2, None, 4, 'both', [0, 3, 4, 0]),
        (0, 2, 3, 'right', [1, 2, 0]),
        (0, 2, 3, 'left', [0, 1, 2]),
        (0, 2, 3, 'both', [1, 2, 0]),
        (0, 2, 4, 'right', [1, 2, 0, 0]),
        (0, 2, 4, 'left', [0, 0, 1, 2]),
        (0, 2, 4, 'both', [0, 1, 2, 0]),
        (2, 0, 3, 'right', [3, 4, 0]),
        (2, 0, 3, 'left', [0, 3, 4]),
        (2, 0, 3, 'both', [3, 4, 0]),
        (2, 0, 4, 'right', [3, 4, 0, 0]),
        (2, 0, 4, 'left', [0, 0, 3, 4]),
        (2, 0, 4, 'both', [0, 3, 4, 0]),
        (1, 1, 3, 'right', [2, 3, 0]),
        (1, 1, 3, 'left', [0, 2, 3]),
        (1, 1, 3, 'both', [2, 3, 0]),
    ]
)
def test_Trim_fill_zeros(
        sampling_rate,
        unit,
        signal,
        fill,
        start_pos,
        end_pos,
        duration,
        fill_pos,
        expected_signal,
):
    transform = Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        fill_pos=fill_pos,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        np.testing.assert_equal(buf._data, np.array(expected_signal))


# Trim, fill='loop'
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('unit', ['samples'])
@pytest.mark.parametrize('signal', [[1, 2, 3, 4]])
@pytest.mark.parametrize('fill', ['loop'])
@pytest.mark.parametrize(
    'start_pos, end_pos, duration, fill_pos, expected_signal',
    [
        (None, None, 2, 'right', [2, 3]),
        (None, None, 2, 'left', [2, 3]),
        (None, None, 2, 'both', [2, 3]),
        (None, None, 3, 'right', [1, 2, 3]),
        (None, None, 3, 'left', [1, 2, 3]),
        (None, None, 3, 'both', [1, 2, 3]),
        (None, None, 6, 'right', [1, 2, 3, 4, 1, 2]),
        (None, None, 6, 'left', [3, 4, 1, 2, 3, 4]),
        (None, None, 6, 'both', [4, 1, 2, 3, 4, 1]),
        (None, None, 8, 'right', [1, 2, 3, 4, 1, 2, 3, 4]),
        (None, None, 8, 'left', [1, 2, 3, 4, 1, 2, 3, 4]),
        (None, None, 8, 'both', [3, 4, 1, 2, 3, 4, 1, 2]),
        (None, 2, 3, 'right', [1, 2, 1]),
        (None, 2, 3, 'left', [2, 1, 2]),
        (None, 2, 3, 'both', [1, 2, 1]),
        (2, None, 3, 'right', [3, 4, 3]),
        (2, None, 3, 'left', [4, 3, 4]),
        (2, None, 3, 'both', [3, 4, 3]),
        (2, None, 4, 'right', [3, 4, 3, 4]),
        (2, None, 4, 'left', [3, 4, 3, 4]),
        (2, None, 4, 'both', [4, 3, 4, 3]),
        (0, 2, 3, 'right', [1, 2, 1]),
        (0, 2, 3, 'left', [2, 1, 2]),
        (0, 2, 3, 'both', [1, 2, 1]),
        (0, 2, 4, 'right', [1, 2, 1, 2]),
        (0, 2, 4, 'left', [1, 2, 1, 2]),
        (0, 2, 4, 'both', [2, 1, 2, 1]),
        (2, 0, 3, 'right', [3, 4, 3]),
        (2, 0, 3, 'left', [4, 3, 4]),
        (2, 0, 3, 'both', [3, 4, 3]),
        (2, 0, 4, 'right', [3, 4, 3, 4]),
        (2, 0, 4, 'left', [3, 4, 3, 4]),
        (2, 0, 4, 'both', [4, 3, 4, 3]),
        (1, 1, 3, 'right', [2, 3, 2]),
        (1, 1, 3, 'left', [3, 2, 3]),
        (1, 1, 3, 'both', [2, 3, 2]),
        (None, 3, 6, 'right', [1, 1, 1, 1, 1, 1]),
        (None, 3, 6, 'left', [1, 1, 1, 1, 1, 1]),
        (None, 3, 6, 'both', [1, 1, 1, 1, 1, 1]),
        (3, None, 6, 'right', [4, 4, 4, 4, 4, 4]),
        (3, None, 6, 'left', [4, 4, 4, 4, 4, 4]),
        (3, None, 6, 'both', [4, 4, 4, 4, 4, 4]),
        (None, 2, 6, 'right', [1, 2, 1, 2, 1, 2]),
        (None, 2, 6, 'left', [1, 2, 1, 2, 1, 2]),
        (None, 2, 6, 'both', [1, 2, 1, 2, 1, 2]),
        (2, None, 6, 'right', [3, 4, 3, 4, 3, 4]),
        (2, None, 6, 'left', [3, 4, 3, 4, 3, 4]),
        (2, None, 6, 'both', [3, 4, 3, 4, 3, 4]),
    ]
)
def test_Trim_fill_loop(
        sampling_rate,
        unit,
        signal,
        fill,
        start_pos,
        end_pos,
        duration,
        fill_pos,
        expected_signal,
):
    transform = Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        fill_pos=fill_pos,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        np.testing.assert_equal(buf._data, np.array(expected_signal))


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('signal', [[1, 2, 3]])
@pytest.mark.parametrize(
    'start_pos, end_pos, duration, unit, error, error_msg',
    [
        (  # negative start_pos
            -1.0, None, None, 'seconds', ValueError,
            "'start_pos' must be >=0.",
        ),
        (  # negative start_pos
            -1, None, None, 'samples', ValueError,
            "'start_pos' must be >=0.",
        ),
        (  # negative end_pos
            None, -1.0, None, 'seconds', ValueError,
            "'end_pos' must be >=0.",
        ),
        (  # negative end_pos
            None, -1, None, 'samples', ValueError,
            "'end_pos' must be >=0.",
        ),
        (  # negative duration
            0, None, -1.0, 'seconds', ValueError,
            "'duration' must be >=0.",
        ),
        (  # negative duration
            0, None, -1, 'samples', ValueError,
            "'duration' must be >=0.",
        ),
        (  # duration too small
            0, None, 0.0001, 'seconds', ValueError,
            "Your combination of "
            "'duration' = 0.0001 seconds "
            "and 'sampling_rate' = 8000 Hz "
            "would lead to an empty buffer "
            "which is forbidden.",
        ),
        (  # start_pos >= len(signal)
            3, None, None, 'samples', ValueError,
            "'start_pos' must be <3.",
        ),
        (  # start_pos >= len(signal)
            3, None, 4, 'samples', ValueError,
            "'start_pos' must be <3.",
        ),
        (  # end_pos >= len(signal)
            None, 3, None, 'samples', ValueError,
            "'end_pos' must be <3.",
        ),
        (  # end_pos >= len(signal)
            None, 3, 4, 'samples', ValueError,
            "'end_pos' must be <3.",
        ),
        (  # start_pos + end_pos >= len(signal)
            1, 2, None, 'samples', ValueError,
            "'start_pos' + 'end_pos' must be <3.",
        ),
        (  # start_pos + end_pos >= len(signal)
            1, 2, 4, 'samples', ValueError,
            "'start_pos' + 'end_pos' must be <3.",
        ),
    ]
)
def test_Trim_error_call(
        sampling_rate,
        signal,
        start_pos,
        end_pos,
        duration,
        unit,
        error,
        error_msg,
):
    with pytest.raises(error, match=re.escape(error_msg)):
        transform = Trim(
            start_pos=start_pos,
            end_pos=end_pos,
            duration=duration,
            unit=unit,
        )
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )

        with AudioBuffer.from_array(signal, sampling_rate) as buf:
            transform(buf)


@pytest.mark.parametrize(
    'fill, fill_pos, error, error_msg',
    [
        (  # wrong fill
            'unknown',
            'right',
            ValueError,
            (
                "Unknown fill strategy 'unknown'. "
                "Supported strategies are: "
                "none, zeros, loop."
            ),
        ),
        (  # wrong fill_pos
            'none',
            'unknown',
            ValueError,
            (
                "Unknown fill_pos 'unknown'. "
                "Supported positions are: "
                "right, left, both."
            ),
        ),
    ],
)
def test_Trim_error_init(
        fill,
        fill_pos,
        error,
        error_msg,
):
    with pytest.raises(error, match=re.escape(error_msg)):
        Trim(
            fill=fill,
            fill_pos=fill_pos,
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
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
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
def test_WhiteNoiseUniform(duration, sampling_rate, gain_db, snr_db):

    seed = 0
    auglib.seed(seed)

    transform = WhiteNoiseUniform(
        gain_db=gain_db,
        snr_db=snr_db,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
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
