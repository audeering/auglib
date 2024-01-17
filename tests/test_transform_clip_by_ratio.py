import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize(
    "ratio, normalize, soft, signal, expected_signal",
    [
        (
            0.1,
            False,
            False,
            np.array([0.0]),
            np.array([0.0]),
        ),
        (
            0,
            False,
            False,
            np.array([1.0]),
            np.array([1.0]),
        ),
        (
            0.5,
            False,
            False,
            np.array([1.0]),
            np.array([1.0]),
        ),
        (
            1.0,
            False,
            False,
            np.array([1.0]),
            np.array([1.0]),
        ),
        (
            0,
            False,
            False,
            np.array([0.5, 2.0]),
            np.array([0.5, 2.0]),
        ),
        (
            0.5,
            False,
            False,
            np.array([0.5, 2.0]),
            np.array([0.5, 0.5]),
        ),
        (
            1.0,
            False,
            True,
            np.array([0.5, 2.0]),
            np.array([0.5, 0.5]),
        ),
        (
            0.5,
            True,
            False,
            np.array([0.5, 2.0]),
            np.array([1.0, 1.0]),
        ),
        (
            1 / 3,
            False,
            False,
            np.array([0.5, 1.0, 2.0]),
            np.array([0.5, 1.0, 1.0]),
        ),
        (
            0.5,
            False,
            False,
            np.array([[0.5, 2.0]]),
            np.array([[0.5, 0.5]]),
        ),
    ],
)
def test_ClipByRatio(
    ratio,
    normalize,
    soft,
    signal,
    expected_signal,
):
    expected_signal = expected_signal.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.ClipByRatio(
        ratio,
        normalize=normalize,
        soft=soft,
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
