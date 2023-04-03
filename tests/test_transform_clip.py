import numpy as np
import pytest

import audobject
import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'threshold, normalize, signal, expected_signal',
    [
        (1.0, False, [-1.5, -0.5, 0.5, 1.5], [-1.0, -0.5, 0.5, 1.0]),
        (0.5, False, [-1.5, 0.5], [-0.5, 0.5]),
        (0.5, True, [-1.5, 0.5], [-1.0, 1.0]),
        (0.5, True, [-1.5, 0.25], [-1.0, 0.5]),
    ],
)
def test_Clip(sampling_rate, threshold, normalize, signal, expected_signal):

    transform = auglib.transform.Clip(
        threshold=auglib.utils.to_db(threshold),
        normalize=normalize,
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
