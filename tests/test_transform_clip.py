import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize(
    "threshold, normalize, signal, expected_signal",
    [
        (
            1.0,
            False,
            np.array([-1.5, -0.5, 0.5, 1.5]),
            np.array([-1.0, -0.5, 0.5, 1.0]),
        ),
        (
            0.5,
            False,
            np.array([-1.5, 0.5]),
            np.array([-0.5, 0.5]),
        ),
        (
            0.5,
            True,
            np.array([-1.5, 0.5]),
            np.array([-1.0, 1.0]),
        ),
        (
            0.5,
            True,
            np.array([-1.5, 0.25]),
            np.array([-1.0, 0.5]),
        ),
    ],
)
def test_Clip(threshold, normalize, signal, expected_signal):
    expected_signal = expected_signal.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.Clip(
        threshold=auglib.utils.to_db(threshold),
        normalize=normalize,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    augmented_signal = transform(signal)
    assert augmented_signal.dtype == expected_signal.dtype
    assert augmented_signal.shape == expected_signal.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected_signal,
        decimal=4,
    )
