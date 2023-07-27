import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[1, 1]])
@pytest.mark.parametrize(
    'duration, unit, value, expected',
    [
        (0, 'samples', 0, [1, 1]),
        (0, 'seconds', 0, [1, 1]),
        (1, 'samples', 2, [1, 1, 2]),
        (2, 'samples', 2, [1, 1, 2, 2]),
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
    transform = auglib.transform.AppendValue(
        duration,
        value,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False)
    )

    with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
        )
