import numpy as np
import pytest

import audmath
import audobject

import auglib


@pytest.mark.parametrize("duration", [1.0])
@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("threshold", [-12])
@pytest.mark.parametrize(
    "signal_peak_db, ratio, makeup_db, clip, expected_peak_db",
    [
        (-3, 20.0, None, False, -3),
        (10, 20.0, None, False, 10),
        (-3, 2.0, 0.0, False, -3),
        (10, 2.0, 0.0, False, 10),
        (-3, 2.0, 10.0, False, 7),
        (10, 2.0, 0.0, True, 0),
        (-3, 2.0, 0.0, True, -3),
        (-3, 2.0, 4.0, True, 0),
    ],
)
def test_compress_dynamic_range(
    duration,
    sampling_rate,
    threshold,
    signal_peak_db,
    ratio,
    makeup_db,
    clip,
    expected_peak_db,
):
    signal = np.zeros((1, int(duration * sampling_rate)))
    tone = auglib.transform.Tone(220.0, shape="square")
    normalize = auglib.transform.NormalizeByPeak(peak_db=signal_peak_db)
    signal = normalize(tone(signal, sampling_rate))

    transform = auglib.transform.CompressDynamicRange(
        threshold,
        ratio,
        attack_time=0.0,
        release_time=0.1,
        knee_radius_db=6.0,
        makeup_db=makeup_db,
        clip=clip,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    augmented_signal = transform(signal, sampling_rate)
    peak = audmath.db(np.max(np.abs(augmented_signal)))
    assert np.isclose(peak, expected_peak_db, atol=1e-02)


@pytest.mark.parametrize(
    "sampling_rate, expected_error, expected_error_msg",
    [
        (
            None,
            ValueError,
            "sampling_rate is 'None', but required.",
        ),
    ],
)
def test_compress_dynamic_range_errors(
    sampling_rate,
    expected_error,
    expected_error_msg,
):
    with pytest.raises(expected_error, match=expected_error_msg):
        transform = auglib.transform.CompressDynamicRange(-12, 1)
        transform(np.ones((1, 4000)), sampling_rate)
