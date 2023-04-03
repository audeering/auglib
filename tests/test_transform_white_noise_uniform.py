import numpy as np
import pytest

import audmath
import audobject
import auglib


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

    transform = auglib.transform.WhiteNoiseUniform(
        gain_db=gain_db,
        snr_db=snr_db,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with transform(auglib.AudioBuffer(duration, sampling_rate)) as noise:
        with auglib.AudioBuffer(
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
            auglib.core.buffer.lib.AudioBuffer_addWhiteNoiseUniform(
                expected_noise._obj,
                gain_db,
            )
            # Double check volume is correct
            np.testing.assert_almost_equal(
                audmath.db(audmath.rms(expected_noise._data), bottom=-140),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_equal(
                noise._data,
                expected_noise._data,
            )
            np.testing.assert_almost_equal(
                audmath.db(audmath.rms(noise._data), bottom=-140),
                expected_volume,
                decimal=1,
            )
            np.testing.assert_almost_equal(
                noise._data.mean(),
                expected_mean,
                decimal=1,
            )
