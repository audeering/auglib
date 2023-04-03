import numpy as np
import pytest

import audobject
import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'ratio, normalize, soft, signal, expected_signal',
    [
        (0.5, False, False, [0.5, 2.0], [0.5, 0.5]),
        (0.5, False, True, [0.5, 2.0], [0.5, 0.5]),
        (0.5, True, False, [0.5, 2.0], [1.0, 1.0]),
        (1 / 3, False, False, [0.5, 1.0, 2.0], [0.5, 1.0, 1.0]),
    ],
)
def test_ClipByRatio(
        sampling_rate,
        ratio,
        normalize,
        soft,
        signal,
        expected_signal,
):

    transform = auglib.transform.ClipByRatio(
        ratio,
        normalize=normalize,
        soft=soft,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        np.testing.assert_almost_equal(
            buf._data,
            np.array(expected_signal, dtype=np.float32),
            decimal=4,
        )
