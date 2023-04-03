import numpy as np
import pytest

import audobject
import auglib


@pytest.mark.parametrize('sampling_rate', [8000, 44100])
@pytest.mark.parametrize('signal', [np.linspace(-0.5, 0.5, num=10)])
def test_Normalize(sampling_rate, signal):

    with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform = auglib.transform.NormalizeByPeak()
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buf)
        assert np.abs(buf._data).max() == 1.0
