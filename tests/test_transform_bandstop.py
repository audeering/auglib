import numpy as np
import pytest
import scipy

import audobject

import auglib


@pytest.mark.parametrize('sampling_rate', [8000, 44100])
@pytest.mark.parametrize('duration', [10])
def test_bandstop(duration, sampling_rate):

    # generate a boxcar signal (step up...step down)
    signal = np.zeros(duration * sampling_rate, dtype='float32')
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
        btype='bandstop',
        fs=sampling_rate,
    )
    expected_signal = scipy.signal.lfilter(b, a, signal)
    expected_signal = expected_signal.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.BandStop(
        center,
        bandwidth,
        sampling_rate=sampling_rate,
        order=order,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    augmented_signal = transform(signal)
    assert augmented_signal.dtype == expected_signal.dtype
    assert augmented_signal.shape == expected_signal.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected_signal,
    )


def test_bandstop_errors():
    with pytest.raises(ValueError):
        design = 'non-supported'
        auglib.transform.BandStop(1, 1, design=design)
