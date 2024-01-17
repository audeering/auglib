import numpy as np
import pytest

import audobject
import audresample

import auglib


@pytest.mark.parametrize(
    "signal, original_rate, target_rate",
    [
        (
            np.random.uniform(-1, 1, (1, 16000)).astype("float32"),
            16000,
            8000,
        ),
        (
            np.random.uniform(-1, 1, (1, 16000)).astype("float32"),
            16000,
            16000,
        ),
        (
            np.random.uniform(-1, 1, (1, 8000)).astype("float32"),
            8000,
            16000,
        ),
    ],
)
def test_resample(signal, original_rate, target_rate):
    expected = audresample.resample(signal, original_rate, target_rate)

    transform = auglib.transform.Resample(
        target_rate,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )
    np.testing.assert_array_equal(
        transform(signal, original_rate),
        expected,
        strict=True,
    )


@pytest.mark.parametrize(
    "sampling_rate, expected_error, expected_error_msg",
    [
        (
            None,
            ValueError,
            "sampling_rate is 'None', but required.",
        ),
    ],
)
def test_resample_errors(
    sampling_rate,
    expected_error,
    expected_error_msg,
):
    with pytest.raises(expected_error, match=expected_error_msg):
        transform = auglib.transform.Resample(8000)
        transform(np.ones((1, 4000)), sampling_rate)


def test_resample_warning():
    expected_warning = (
        "'override' argument is ignored " "and will be removed with version 1.2.0."
    )
    with pytest.warns(UserWarning, match=expected_warning):
        auglib.transform.Resample(8000, override=True)
