import numpy as np
import pytest
import scipy

import audobject

import auglib


@pytest.mark.parametrize('sampling_rate', [8000, 44100])
@pytest.mark.parametrize('duration', [10])
@pytest.mark.parametrize('order', [1, 4])
@pytest.mark.parametrize('cutoff', [1000, 3000])
def test_highpass(sampling_rate, duration, order, cutoff):

    # generate a boxcar signal (step up...step down)
    signal = np.zeros((1, int(duration * sampling_rate)))
    start = int(duration * sampling_rate / 4)
    end = int(duration * sampling_rate * 3 / 4)
    signal[start:end] = 1.0

    # Highpass
    b, a = scipy.signal.butter(
        order,
        cutoff,
        'highpass',
        fs=sampling_rate,
    )
    expected = scipy.signal.lfilter(b, a, signal)
    expected = expected.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.HighPass(
        cutoff,
        order=order,
        sampling_rate=sampling_rate,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    augmented_signal = transform(signal)
    assert augmented_signal.dtype == expected.dtype
    assert augmented_signal.shape == expected.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected,
    )


def test_highpass_errors():
    with pytest.raises(ValueError):
        design = 'non-supported'
        auglib.transform.HighPass(1, design=design)
