import numpy as np
import pytest
import scipy

import audobject

import auglib


@pytest.mark.parametrize('sr', [8000, 44100])
@pytest.mark.parametrize('n', [10])
def test_lowpass(n, sr):

    # generate a boxcar signal (step up...step down)
    sig_in = np.zeros(n * sr, dtype='float32')
    sig_in[int(n * sr / 4):int(n * sr * 3 / 4)] = 1.0

    # Lowpass
    b, a = scipy.signal.butter(1, 0.5, 'lowpass')
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    transform = auglib.transform.LowPass(0.5 * sr / 2)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        transform(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)


def test_lowpass_errors():
    with pytest.raises(ValueError):
        design = 'non-supported'
        auglib.transform.LowPass(1, design=design)
