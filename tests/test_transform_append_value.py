import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'base, duration, unit, value, expected',
    [
        ([[1, 1]], 0, 'samples', 0, [[1, 1]]),
        ([[1, 1]], 0, 'seconds', 0, [[1, 1]]),
        ([[1, 1]], 1, 'samples', 2, [[1, 1, 2]]),
        ([[1, 1]], 2, 'samples', 2, [[1, 1, 2, 2]]),
        ([1, 1], 0, 'samples', 0, [1, 1]),
        ([1, 1], 0, 'seconds', 0, [1, 1]),
        ([1, 1], 1, 'samples', 2, [1, 1, 2]),
        ([1, 1], 2, 'samples', 2, [1, 1, 2, 2]),
    ],
)
def test_AppendValue(
        sampling_rate,
        base,
        duration,
        unit,
        value,
        expected,
):
    base = np.array(base)
    expected = np.array(expected, dtype=auglib.core.transform.DTYPE)

    transform = auglib.transform.AppendValue(
        duration,
        value,
        unit=unit,
        sampling_rate=sampling_rate,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False)
    )

    np.testing.assert_array_equal(
        transform(base),
        expected,
        strict=True,  # ensure same shape and dtype
    )
