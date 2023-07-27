import numpy as np
import pytest

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

    with auglib.AudioBuffer.read(original_wav) as buf:
        transform(buf)
        result = buf._data
        with auglib.AudioBuffer.read(target_wav) as target_buf:
            target = target_buf._data
            length = min(result.size, target.size)
            np.testing.assert_allclose(
                result[:length], target[:length], rtol=0.0, atol=0.065
            )
