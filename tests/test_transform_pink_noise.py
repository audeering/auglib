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
def test_PinkNoise(duration, sampling_rate, gain_db, snr_db):
    seed = 0
    auglib.seed(seed)
    transform = auglib.transform.PinkNoise(
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

            # Get the empiric measure of the RMS energy for Pink Noise
            with auglib.AudioBuffer(
                    len(noise), noise.sampling_rate, unit='samples'
            ) as tmp_pink_noise:
                auglib.seed(seed)
                auglib.core.buffer.lib.AudioBuffer_addPinkNoise(
                    tmp_pink_noise._obj,
                    0,
                )
                noise_volume = audmath.db(audmath.rms(tmp_pink_noise._data))

                if snr_db is not None:
                    gain_db = -120 - snr_db - noise_volume

                expected_volume = noise_volume + gain_db

                if snr_db is not None:
                    auglib.core.buffer.lib.AudioBuffer_mix(
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
                    auglib.core.buffer.lib.AudioBuffer_addPinkNoise(
                        expected_noise._obj,
                        gain_db,
                    )

            # Check volume is correct
            np.testing.assert_almost_equal(
                audmath.db(audmath.rms(expected_noise._data), bottom=-140),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_almost_equal(
                audmath.db(audmath.rms(noise._data), bottom=-140),
                expected_volume,
                decimal=1,
            )

            np.testing.assert_equal(
                noise._data,
                expected_noise._data,
            )
