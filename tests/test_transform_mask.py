import numpy as np
import pytest

import audobject

import auglib


@pytest.mark.parametrize(
    "signal, sampling_rate, transform",
    [
        (
            np.zeros((1, 10)),
            8000,
            auglib.transform.Function(lambda x, sr: x + 1),
        )
    ],
)
@pytest.mark.parametrize(
    "start_pos, duration, step, invert, expected",
    [
        (0, None, None, False, np.zeros((1, 10))),
        (0, None, None, True, np.ones((1, 10))),
        (0, 0, None, False, np.ones((1, 10))),
        (0, 0, None, True, np.zeros((1, 10))),
        (0, 5, None, False, [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]),
        (0, 5, None, True, [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
        (3, 5, None, False, [[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]]),
        (3, 5, None, True, [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        (3, None, None, False, [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]),
        (3, None, None, True, [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
        (3, 0, None, False, np.ones((1, 10))),
        (3, 0, None, True, np.zeros((1, 10))),
        (0, None, 2, False, [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]),
        (0, None, 2, True, [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]),
        (0, 0, 2, False, np.ones((1, 10))),
        (0, 0, 2, True, np.zeros((1, 10))),
        (3, 5, 2, False, [[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]]),
        (3, 5, 2, True, [[0, 0, 0, 1, 1, 0, 0, 1, 0, 0]]),
        (0, None, (1, 3), False, [[0, 1, 1, 1, 0, 1, 1, 1, 0, 1]]),
        (0, None, (1, 3), True, [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0]]),
        (0, 0, (1, 3), False, np.ones((1, 10))),
        (0, 0, (1, 3), True, np.zeros((1, 10))),
        (3, 5, (1, 3), False, [[1, 1, 1, 0, 1, 1, 1, 0, 1, 1]]),
        (3, 5, (1, 3), True, [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]),
    ],
)
def test_mask(
    signal,
    sampling_rate,
    transform,
    start_pos,
    duration,
    step,
    invert,
    expected,
):
    if isinstance(expected, np.ndarray):
        expected = expected.astype(auglib.core.transform.DTYPE)
    else:
        expected = np.array(
            expected,
            dtype=auglib.core.transform.DTYPE,
        )
    mask = auglib.transform.Mask(
        transform,
        start_pos=start_pos,
        duration=duration,
        step=step,
        invert=invert,
        unit="samples",
    )
    mask = audobject.from_yaml_s(
        mask.to_yaml_s(include_version=False),
    )

    np.testing.assert_array_equal(mask(signal), expected, strict=True)
