import numpy as np
import pytest

import audobject
import audresample

import auglib


@pytest.mark.parametrize(
    'signal, original_rate, target_rate',
    [
        (
            np.random.uniform(-1, 1, (1, 16000)).astype(np.float32),
            16000,
            8000,
        ),
        (
            np.random.uniform(-1, 1, (1, 16000)).astype(np.float32),
            16000,
            16000,
        ),
        (
            np.random.uniform(-1, 1, (1, 8000)).astype(np.float32),
            8000,
            16000,
        ),
    ]
)
@pytest.mark.parametrize(
    'override',
    [False, True],
)
def test_Resample(signal, original_rate, target_rate, override):

    expected = audresample.resample(signal, original_rate, target_rate)

    transform = auglib.transform.Resample(target_rate, override=override)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(signal, original_rate) as buf:
        transform(buf)
        resampled = buf.to_array()
        if override:
            assert buf.sampling_rate == target_rate
        else:
            assert buf.sampling_rate == original_rate

    np.testing.assert_equal(resampled, expected)
