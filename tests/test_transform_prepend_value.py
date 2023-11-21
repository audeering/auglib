import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [np.array([1, 1])])
@pytest.mark.parametrize(
    'duration, unit, value, expected',
    [
        (0, 'samples', 0, np.array([1, 1])),
        (0, 'seconds', 0, np.array([1, 1])),
        (1, 'samples', 2, np.array([2, 1, 1])),
        (2, 'samples', 2, np.array([2, 2, 1, 1])),
    ],
)
def test_prepend_value(
        sampling_rate,
        base,
        duration,
        unit,
        value,
        expected,
):
    expected = expected.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.PrependValue(
        duration,
        value,
        sampling_rate=sampling_rate,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )
    np.testing.assert_array_equal(
        transform(base),
        expected,
        strict=True,
    )
