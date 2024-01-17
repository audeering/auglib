import numpy as np
import pytest

import auglib


@pytest.mark.parametrize(
    "keep_tail, impulse_response, signal, expected",
    [
        (
            False,
            np.array([1, 0, 0]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ),
        (
            False,
            np.array([0, 1, 0]),
            np.array([1, 2, 3]),
            np.array([0, 1, 2]),
        ),
        (
            True,
            np.array([0, 1, 0]),
            np.array([1, 2, 3]),
            np.array([0, 1, 2, 3, 0]),
        ),
        (
            True,
            np.array([0, 1, 0, 0]),
            np.array([1, 2, 3]),
            np.array([0, 1, 2, 3, 0, 0]),
        ),
    ],
)
def test_FFTConvolve(keep_tail, impulse_response, signal, expected):
    expected = expected.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.FFTConvolve(
        impulse_response,
        keep_tail=keep_tail,
    )
    error_msg = (
        "Cannot serialize an instance "
        "of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and pass filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    augmented_signal = transform(signal)
    assert augmented_signal.dtype == expected.dtype
    assert augmented_signal.shape == expected.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected,
        decimal=4,
    )
