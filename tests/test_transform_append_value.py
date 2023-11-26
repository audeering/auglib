import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'signal, duration, unit, value, expected',
    [
        ([[1, 1]], 0, 'samples', 0, [[1, 1]]),
        ([[1, 1]], 0, 'seconds', 0, [[1, 1]]),
        ([[1, 1]], 1, 'samples', 2, [[1, 1, 2]]),
        ([[1, 1]], 2, 'samples', 2, [[1, 1, 2, 2]]),
        ([1, 1], 0, 'samples', 0, [1, 1]),
        ([1, 1], 0, 'seconds', 0, [1, 1]),
        ([1, 1], 1, 'samples', 2, [1, 1, 2]),
        ([1, 1], 2, 'samples', 2, [1, 1, 2, 2]),
        ([1, 1], 0, 'relative', 2, [1, 1]),
        ([1, 1], 0.5, 'relative', 2, [1, 1, 2]),
        ([1, 1], 1, 'relative', 2, [1, 1, 2, 2]),
        ([1, 1], 1.5, 'relative', 2, [1, 1, 2, 2, 2]),
    ],
)
def test_AppendValue(
        sampling_rate,
        signal,
        duration,
        unit,
        value,
        expected,
):
    signal = np.array(signal)
    expected = np.array(expected, dtype=auglib.core.transform.DTYPE)

    transform = auglib.transform.AppendValue(
        duration,
        value,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False)
    )

    np.testing.assert_array_equal(
        transform(signal, sampling_rate),
        expected,
        strict=True,  # ensure same shape and dtype
    )
