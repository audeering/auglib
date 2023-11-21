import numpy as np
import pytest

import audiofile
import audobject

import auglib


@pytest.mark.parametrize(
    'bit_rate', [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]
)
def test_AMRNB(bit_rate):

    original_wav = 'tests/test-assets/opensmile.wav'
    target_wav = f'tests/test-assets/opensmile_amrnb_rate{bit_rate}_ffmpeg.wav'

    transform = auglib.transform.AMRNB(bit_rate)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    signal, _ = audiofile.read(original_wav, always_2d=True)
    expected_signal, _ = audiofile.read(target_wav, always_2d=True)
    length = min(signal.shape[1], expected_signal.shape[1])
    np.testing.assert_allclose(
        transform(signal)[:, :length],
        expected_signal[:, :length],
        rtol=0.0,
        atol=0.065,
    )
