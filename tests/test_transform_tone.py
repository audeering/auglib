import numpy as np
import pytest
import scipy

import audmath
import audobject

import auglib


@pytest.mark.parametrize("duration", [1.0])
@pytest.mark.parametrize("sampling_rate", [96000])
@pytest.mark.parametrize("frequency", [1000])
@pytest.mark.parametrize("shape", ["sine", "square", "triangle", "sawtooth"])
@pytest.mark.parametrize(
    "gain_db, snr_db",
    [
        (-10, None),
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ],
)
def test_tone(duration, sampling_rate, frequency, shape, gain_db, snr_db):
    transform = auglib.transform.Tone(
        frequency,
        shape=shape,
        gain_db=gain_db,
        snr_db=snr_db,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    signal = np.zeros((1, int(duration * sampling_rate)))

    if gain_db is None:
        gain_db = 0.0

    # Expected root mean square values,
    # see https://en.wikipedia.org/wiki/Root_mean_square
    time = np.arange(duration * sampling_rate, dtype=float) / sampling_rate
    omega = 2 * np.pi * frequency
    if shape == "sine":
        expected_tone = np.sin(omega * time)
        expected_rms_db = audmath.db(1 / np.sqrt(2))
    elif shape == "square":
        expected_tone = -1 * scipy.signal.square(omega * time, duty=0.5)
        expected_rms_db = audmath.db(1)
    elif shape == "triangle":
        expected_tone = -1 * scipy.signal.sawtooth(omega * time, width=0.5)
        expected_rms_db = audmath.db(1 / np.sqrt(3))
    elif shape == "sawtooth":
        expected_tone = scipy.signal.sawtooth(omega * time, width=1)
        expected_rms_db = audmath.db(1 / np.sqrt(3))
    # Check reference signal has expected RMS value
    np.testing.assert_almost_equal(
        audmath.db(audmath.rms(expected_tone)),
        expected_rms_db,
        decimal=2,
    )

    if snr_db is not None:
        # We add noise to an empty signal,
        # which is limited to -120 dB
        gain_db = -120 - snr_db - audmath.db(audmath.rms(expected_tone))

    expected_volume = audmath.db(audmath.rms(expected_tone), bottom=-140) + gain_db

    # Add gain to expected tone
    gain = 10 ** (gain_db / 20)
    expected_tone *= gain

    tone = transform(signal, sampling_rate)

    # We cannot use np.testing.assert_almost_equal()
    # for comparing square, triangle, and sawtooth signals
    # as there might be a shift of one sample
    # from time to time,
    # which would let the test fail.
    assert tone.dtype == auglib.core.transform.DTYPE
    assert np.mean(np.abs(tone - expected_tone)) < 1e4
    np.testing.assert_almost_equal(
        audmath.db(audmath.rms(tone), bottom=-140),
        expected_volume,
        decimal=2,
    )


@pytest.mark.parametrize(
    "shape, sampling_rate, expected_error, expected_error_msg",
    [
        (
            "non-supported",
            16000,
            ValueError,
            (
                "Unknown tone shape 'non-supported'. "
                "Supported shapes are: sine, square, triangle, sawtooth."
            ),
        ),
        (
            "sine",
            None,
            ValueError,
            "sampling_rate is 'None', but required.",
        ),
    ],
)
def test_tone_errors(
    shape,
    sampling_rate,
    expected_error,
    expected_error_msg,
):
    with pytest.raises(expected_error, match=expected_error_msg):
        transform = auglib.transform.Tone(440, shape=shape)
        transform(np.ones((1, 4000)), sampling_rate)
