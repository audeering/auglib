import numpy as np
import pytest

import audmath
import audobject

import auglib


@pytest.mark.parametrize("duration", [1.0])
@pytest.mark.parametrize("sampling_rate", [8000])
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
def test_pink_noise(duration, sampling_rate, gain_db, snr_db):
    seed = 0
    auglib.seed(seed)
    transform = auglib.transform.PinkNoise(
        gain_db=gain_db,
        snr_db=snr_db,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    base = np.zeros((1, int(duration * sampling_rate)))

    if gain_db is None:
        gain_db = 0.0

    # Get the empiric measure of the RMS energy for Pink Noise
    auglib.seed(seed)
    pink_noise = auglib.transform.PinkNoise()(base)
    noise_volume = audmath.db(audmath.rms(pink_noise))

    if snr_db is not None:
        gain_db = -120 - snr_db - noise_volume

    expected_volume = noise_volume + gain_db
    expected_noise = audmath.inverse_db(gain_db) * pink_noise

    # Check volume is correct
    np.testing.assert_almost_equal(
        audmath.db(audmath.rms(expected_noise), bottom=-140),
        expected_volume,
        decimal=1,
    )

    auglib.seed(seed)
    noise = transform(base)
    expected_noise = expected_noise.astype(auglib.core.transform.DTYPE)

    if gain_db <= 0:
        assert np.min(noise).round(4) >= -1 and np.min(noise) < 0
        assert np.max(noise).round(4) <= 1 and np.max(noise) > 0
    else:
        assert np.min(noise) < -1
        assert np.max(noise) > 1

    np.testing.assert_almost_equal(
        audmath.db(audmath.rms(noise), bottom=-140),
        expected_volume,
        decimal=1,
    )
    assert noise.shape == expected_noise.shape
    assert noise.dtype == expected_noise.dtype
    np.testing.assert_almost_equal(noise, expected_noise, decimal=6)


@pytest.mark.parametrize(
    "signal",
    [
        # Odd,
        # compare https://github.com/audeering/auglib/issues/23
        np.ones((1, 30045)),
        # Even
        np.ones((1, 200)),
    ],
)
def test_pink_noise_odd_and_even_samples(signal):
    transform = auglib.transform.PinkNoise()
    augmented_signal = transform(signal)
    assert signal.shape == augmented_signal.shape
