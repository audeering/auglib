import numpy as np
import pytest
import scipy

import audobject
import auglib


@pytest.mark.parametrize('sr', [8000, 44100])
@pytest.mark.parametrize('n', [10])
def test_bandpass(n, sr):

    # generate a boxcar signal (step up...step down)
    sig_in = np.zeros(n * sr, dtype='float32')
    sig_in[int(n * sr / 4):int(n * sr * 3 / 4)] = 1.0

    # Bandpass
    b, a = scipy.signal.butter(
        1,
        np.array([0.5 - 0.25, 0.5 + 0.25]),
        'bandpass',
    )
    sig_out = scipy.signal.lfilter(b, a, sig_in)

    transform = auglib.transform.BandPass(0.5 * sr / 2, 0.5 * sr / 2)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(sig_in, sampling_rate=sr) as buf:
        transform(buf)
        np.testing.assert_almost_equal(buf._data, sig_out)


def test_basndpass_errors():
    with pytest.raises(ValueError):
        design = 'non-supported'
        auglib.transform.BandPass(1, 1, design=design)
