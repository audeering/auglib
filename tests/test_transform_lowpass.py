import numpy as np
import pytest
import scipy

import audobject

import auglib


@pytest.mark.parametrize("sampling_rate", [8000, 44100])
@pytest.mark.parametrize("duration", [10])
@pytest.mark.parametrize("order", [1, 4])
@pytest.mark.parametrize("cutoff", [1000, 3000])
def test_lowpass(duration, sampling_rate, order, cutoff):
    # generate a boxcar signal (step up...step down)
    signal = np.zeros((1, int(duration * sampling_rate)))
    start = int(duration * sampling_rate / 4)
    end = int(duration * sampling_rate * 3 / 4)
    signal[start:end] = 1.0

    # Lowpass
    b, a = scipy.signal.butter(
        order,
        cutoff,
        "lowpass",
        fs=sampling_rate,
    )
    expected = scipy.signal.lfilter(b, a, signal)
    expected = expected.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.LowPass(
        cutoff,
        order=order,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    augmented_signal = transform(signal, sampling_rate)
    assert augmented_signal.dtype == expected.dtype
    assert augmented_signal.shape == expected.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected,
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
        transform = auglib.transform.LowPass(1, design=design)
        transform(np.ones((1, 4000)), sampling_rate)
