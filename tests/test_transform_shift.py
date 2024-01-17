import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize(
    "duration, unit, signal, expected",
    [
        (None, "samples", [1, 2, 3], [1, 2, 3]),
        (0, "samples", [1, 2, 3], [1, 2, 3]),
        (1, "samples", [1, 2, 3], [2, 3, 1]),
        (2, "samples", [1, 2, 3], [3, 1, 2]),
        (3, "samples", [1, 2, 3], [1, 2, 3]),
        (4, "samples", [1, 2, 3], [2, 3, 1]),
        (1.0 / 3, "relative", [1, 2, 3], [2, 3, 1]),
        (2.0 / 3, "relative", [1, 2, 3], [3, 1, 2]),
        (3.0 / 3, "relative", [1, 2, 3], [1, 2, 3]),
        (4.0 / 3, "relative", [1, 2, 3], [2, 3, 1]),
    ],
)
def test_shift(sampling_rate, duration, unit, signal, expected):
    signal = np.array(signal)
    expected = np.array(
        expected,
        dtype=auglib.core.transform.DTYPE,
    )
    transform = auglib.transform.Shift(
        duration=duration,
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
