import numpy as np
import pytest

import audobject
import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'duration, unit, base, expected',
    [
        (None, 'samples', [1, 2, 3], [1, 2, 3]),
        (0, 'samples', [1, 2, 3], [1, 2, 3]),
        (1, 'samples', [1, 2, 3], [2, 3, 1]),
        (2, 'samples', [1, 2, 3], [3, 1, 2]),
        (3, 'samples', [1, 2, 3], [1, 2, 3]),
        (4, 'samples', [1, 2, 3], [2, 3, 1]),
    ],
)
def test_Shift(sampling_rate, duration, unit, base, expected):

    transform = auglib.transform.Shift(duration=duration, unit=unit)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
        )
