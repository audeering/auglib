import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize("signal", [np.array([1, 1])])
@pytest.mark.parametrize(
    "duration, unit, value, sampling_rate, expected",
    [
        (0, "samples", 0, None, np.array([1, 1])),
        (0, "samples", 0, 8000, np.array([1, 1])),
        (0, "seconds", 0, 8000, np.array([1, 1])),
        (auglib.observe.List([0]), "samples", 0, None, np.array([1, 1])),
        (auglib.observe.List([0]), "seconds", 0, 8000, np.array([1, 1])),
        (1, "samples", 2, 8000, np.array([2, 1, 1])),
        (auglib.observe.List([1]), "samples", 2, 8000, np.array([2, 1, 1])),
        (2, "samples", 2, 8000, np.array([2, 2, 1, 1])),
        (0, "relative", 2, 8000, np.array([1, 1])),
        (0.5, "relative", 2, 8000, np.array([2, 1, 1])),
        (1, "relative", 2, 8000, np.array([2, 2, 1, 1])),
        (1.5, "relative", 2, 8000, np.array([2, 2, 2, 1, 1])),
    ],
)
def test_prepend_value(
    sampling_rate,
    signal,
    duration,
    unit,
    value,
    expected,
):
    expected = expected.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.PrependValue(
        duration,
        value,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )
    np.testing.assert_array_equal(
        transform(signal, sampling_rate),
        expected,
        strict=True,
    )
