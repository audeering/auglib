import numpy as np
import pytest

import audmath
import audobject

import auglib


@pytest.mark.parametrize("duration", [1.0])
@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("stddev", [0.1, 0.2, 0.3, 0.4])
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
def test_white_noise_gaussian(
    duration,
    sampling_rate,
    stddev,
    gain_db,
    snr_db,
):
    seed = 0
    auglib.seed(seed)

    transform = auglib.transform.WhiteNoiseGaussian(
        stddev=stddev,
        gain_db=gain_db,
        snr_db=snr_db,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    samples = int(duration * sampling_rate)
    signal = np.zeros((1, samples))

    if gain_db is None:
        gain_db = 0.0

    # Power of white noise is given by std^2
    noise_volume = 10 * np.log10(stddev**2)

    if snr_db is not None:
        # We add noise to an empty signal,
        # which is limited to -120 dB
        gain_db = -120 - snr_db - noise_volume

    expected_volume = noise_volume + gain_db
    expected_mean = 0
    expected_stddev = audmath.inverse_db(gain_db, bottom=-140) * stddev

    # Create expected noise signal
    noise_generator = np.random.default_rng(seed=seed)
    expected_noise = noise_generator.normal(
        expected_mean,
        expected_stddev,
        signal.shape,
    )
    # Double check volume is correct
    np.testing.assert_almost_equal(
        audmath.db(audmath.rms(expected_noise), bottom=-140),
        expected_volume,
        decimal=1,
    )

    auglib.seed(seed)
    noise = transform(signal)
    assert noise.dtype == auglib.core.transform.DTYPE
    np.testing.assert_almost_equal(
        noise,
        expected_noise,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        audmath.db(audmath.rms(noise), bottom=-140),
        expected_volume,
        decimal=1,
    )
    np.testing.assert_almost_equal(
        noise.mean(),
        expected_mean,
        decimal=1,
    )
    np.testing.assert_almost_equal(
        noise.std(),
        expected_stddev,
        decimal=1,
    )
