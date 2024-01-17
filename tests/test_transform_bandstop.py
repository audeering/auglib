import numpy as np
import pytest
import scipy

import audobject

import auglib


@pytest.mark.parametrize("sampling_rate", [8000, 44100])
@pytest.mark.parametrize("duration", [10])
def test_bandstop(duration, sampling_rate):
    # generate a boxcar signal (step up...step down)
    signal = np.zeros(duration * sampling_rate, dtype="float32")
    start = int(duration * sampling_rate * 1 / 4)
    end = int(duration * sampling_rate * 3 / 4)
    signal[start:end] = 1.0

    center = 1000
    bandwidth = 10
    order = 1

    # Expected bandstop
    lowcut = center - bandwidth / 2
    highcut = center + bandwidth / 2
    b, a = scipy.signal.butter(
        order,
        [lowcut, highcut],
        btype="bandstop",
        fs=sampling_rate,
    )
    expected_signal = scipy.signal.lfilter(b, a, signal)
    expected_signal = expected_signal.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.BandStop(
        center,
        bandwidth,
        order=order,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    augmented_signal = transform(signal, sampling_rate)
    assert augmented_signal.dtype == expected_signal.dtype
    assert augmented_signal.shape == expected_signal.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected_signal,
    )


@pytest.mark.parametrize(
    "design, sampling_rate, expected_error, expected_error_msg",
    [
        (
            "non-supported",
            16000,
            ValueError,
            (
                "Unknown filter design 'non-supported'. "
                "Supported designs are: butter."
            ),
        ),
        (
            "butter",
            None,
            ValueError,
            "sampling_rate is 'None', but required.",
        ),
    ],
)
def test_bandpass_errors(
    design,
    sampling_rate,
    expected_error,
    expected_error_msg,
):
    with pytest.raises(expected_error, match=expected_error_msg):
        transform = auglib.transform.BandStop(1, 1, design=design)
        transform(np.ones((1, 4000)), sampling_rate)
