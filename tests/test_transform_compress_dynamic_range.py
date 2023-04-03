import numpy as np
import pytest

import audobject
import auglib


@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('sampling_rate', [8000])
def test_Compression(duration, sampling_rate):
    transform = auglib.transform.CompressDynamicRange(
        -12.0,
        20.0,
        attack_time=0.0,
        release_time=0.1,
        knee_radius_db=6.0,
        makeup_db=None,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer(duration, sampling_rate) as buf:
        auglib.transform.Tone(220.0, shape='square')(buf)
        auglib.transform.NormalizeByPeak(peak_db=-3.0)(buf)
        transform(buf)
        peak1 = buf.peak_db

    transform = auglib.transform.CompressDynamicRange(
        -12.0,
        20.0,
        attack_time=0.0,
        release_time=0.1,
        knee_radius_db=6.0,
        makeup_db=0.0,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer(duration, sampling_rate) as buf:
        auglib.transform.Tone(220.0, shape='square')(buf)
        auglib.transform.NormalizeByPeak(peak_db=-3.0)(buf)
        transform(buf)
        peak2 = buf.peak_db

    assert peak1 > peak2
    assert np.isclose(peak1, -3.0)
