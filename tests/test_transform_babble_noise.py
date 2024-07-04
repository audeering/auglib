import numpy as np
import pytest

import audmath

import auglib


# Test gain and SNR for BabbleNoise
@pytest.mark.parametrize("duration", [0.01, 1.0])
@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("num_speakers", [1, 3])
@pytest.mark.parametrize(
    "gain_db, snr_db",
    [
        (0, None),
        (-10, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ],
)
def test_babble_noise_1(
    duration,
    sampling_rate,
    num_speakers,
    gain_db,
    snr_db,
):
    auglib.seed(0)

    speech = np.ones((1, int(duration * sampling_rate)))

    transform = auglib.transform.BabbleNoise(
        [speech],
        num_speakers=num_speakers,
        gain_db=gain_db,
        snr_db=snr_db,
    )

    signal = np.zeros((1, int(duration * sampling_rate)))

    if snr_db is not None:
        gain_db = -120 - snr_db
    gain = audmath.inverse_db(gain_db)
    expected_babble = gain * np.ones((1, int(duration * sampling_rate)))
    expected_babble = expected_babble.astype(auglib.core.transform.DTYPE)

    babble = transform(signal)
    assert babble.dtype == expected_babble.dtype
    np.testing.assert_almost_equal(
        babble,
        expected_babble,
        decimal=4,
    )


# Test shorter speech signals for BabbleNoise
@pytest.mark.parametrize(
    # NOTE: expected signal depends on seed
    "signal, speech, num_speakers, expected_babble",
    [
        (
            [0, 0, 0],
            [
                [1, 0],
            ],
            1,
            [0, 1, 0],
        ),
        (
            [0, 0, 0, 0],
            [
                [1, 0],
            ],
            1,
            [1, 0, 1, 0],
        ),
        (
            [0, 0, 0],
            [
                [1, 0, 0],
            ],
            1,
            [0, 0, 1],
        ),
        (
            [0, 0, 0],
            [
                [0, 1],
            ],
            3,
            [2 / 3, 1 / 3, 2 / 3],
        ),
        (
            [0, 0, 0],
            [
                [[1, 0]],
            ],
            1,
            [0, 1, 0],
        ),
        (
            [0, 0, 0, 0],
            [
                [[1, 0]],
            ],
            1,
            [1, 0, 1, 0],
        ),
        (
            [0, 0, 0],
            [
                [[1, 0, 0]],
            ],
            1,
            [0, 0, 1],
        ),
        (
            [0, 0, 0],
            [
                [[0, 1]],
            ],
            3,
            [2 / 3, 1 / 3, 2 / 3],
        ),
        (
            [[0, 0, 0]],
            [
                [1, 0],
            ],
            1,
            [[0, 1, 0]],
        ),
        (
            [[0, 0, 0, 0]],
            [
                [1, 0],
            ],
            1,
            [[1, 0, 1, 0]],
        ),
        (
            [[0, 0, 0]],
            [
                [1, 0, 0],
            ],
            1,
            [[0, 0, 1]],
        ),
        (
            [[0, 0, 0]],
            [
                [0, 1],
            ],
            3,
            [[2 / 3, 1 / 3, 2 / 3]],
        ),
        (
            [[0, 0, 0]],
            [
                [[1, 0]],
            ],
            1,
            [[0, 1, 0]],
        ),
        (
            [[0, 0, 0, 0]],
            [
                [[1, 0]],
            ],
            1,
            [[1, 0, 1, 0]],
        ),
        (
            [[0, 0, 0]],
            [
                [[1, 0, 0]],
            ],
            1,
            [[0, 0, 1]],
        ),
        (
            [[0, 0, 0]],
            [
                [[0, 1]],
            ],
            3,
            [[2 / 3, 1 / 3, 2 / 3]],
        ),
    ],
)
def test_babble_noise_2(
    signal,
    speech,
    num_speakers,
    expected_babble,
):
    seed = 0
    auglib.seed(seed)
    signal = np.array(signal)
    speech = [np.array(s) for s in speech]
    expected_babble = np.array(
        expected_babble,
        dtype=auglib.core.transform.DTYPE,
    )

    transform = auglib.transform.BabbleNoise(
        speech,
        num_speakers=num_speakers,
    )

    babble = transform(signal)
    assert babble.dtype == expected_babble.dtype
    np.testing.assert_almost_equal(
        babble,
        expected_babble,
        decimal=4,
    )


@pytest.mark.parametrize(
    "speech, expected_error, expected_error_msg",
    [
        (
            [np.array([0, 1])],
            ValueError,
            "Cannot serialize list "
            "if it contains an instance of <class 'numpy.ndarray'>. "
            "As a workaround, save signal to disk and add filename.",
        ),
        (
            [np.array([[0, 1]])],
            ValueError,
            "Cannot serialize list "
            "if it contains an instance of <class 'numpy.ndarray'>. "
            "As a workaround, save signal to disk and add filename.",
        ),
        (
            [np.array([[0, 1]]), np.array([[0, 1]])],
            ValueError,
            "Cannot serialize list "
            "if it contains an instance of <class 'numpy.ndarray'>. "
            "As a workaround, save signal to disk and add filename.",
        ),
    ],
)
def test_babble_noise_error(speech, expected_error, expected_error_msg):
    with pytest.raises(expected_error, match=expected_error_msg):
        transform = auglib.transform.BabbleNoise(speech)
        transform.to_yaml_s(include_version=False)
