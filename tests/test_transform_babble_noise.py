import numpy as np
import pytest

import audmath
import audobject
import auglib


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

    with auglib.AudioBuffer(
            duration,
            sampling_rate,
            value=1,
    ) as speech:

        transform = auglib.transform.BabbleNoise(
            [speech],
            num_speakers=num_speakers,
            gain_db=gain_db,
            snr_db=snr_db,
        )
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )

        with transform(auglib.AudioBuffer(duration, sampling_rate)) as noise:

            if snr_db is not None:
                gain_db = -120 - snr_db
            gain = audmath.inverse_db(gain_db)
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

    speech_bufs = [
        auglib.AudioBuffer.from_array(s, sampling_rate)
        for s in speech
    ]
    transform = auglib.transform.BabbleNoise(
        speech_bufs,
        num_speakers=num_speakers,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer(duration, sampling_rate, unit='samples') as noise:
        transform(noise)
        np.testing.assert_almost_equal(
            noise._data,
            np.array(expected_noise, dtype=np.float32),
            decimal=4,
        )
    for buf in speech_bufs:
        buf.free()
