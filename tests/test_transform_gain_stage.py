import numpy as np
import pytest

import audmath
import audobject

import auglib


@pytest.mark.parametrize("signal", [np.random.uniform(-0.1, 1.0, 10)])
@pytest.mark.parametrize("gain", [20.0])
@pytest.mark.parametrize("max_peak", [None, 10.0])
@pytest.mark.parametrize("clip", [False, True])
def test_GainStage(signal, gain, max_peak, clip):
    transform = auglib.transform.GainStage(
        gain,
        max_peak_db=max_peak,
        clip=clip,
    )

    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    gain_signal = transform(signal)

    assert gain_signal.dtype == auglib.core.transform.DTYPE
    if clip:
        assert np.abs(gain_signal).max() <= 1.0
    elif max_peak is not None:
        assert np.isclose(
            np.abs(gain_signal).max(),
            audmath.inverse_db(max_peak),
        )
    else:
        assert np.isclose(
            np.abs(gain_signal).max(),
            audmath.inverse_db(gain) * np.abs(signal).max(),
        )
