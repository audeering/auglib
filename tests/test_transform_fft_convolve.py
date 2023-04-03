import numpy as np
import pytest

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'keep_tail, aux, base, expected',
    [
        (False, [1, 0, 0], [1, 2, 3], [1, 2, 3]),
        (False, [0, 1, 0], [1, 2, 3], [0, 1, 2]),
        (True, [0, 1, 0], [1, 2, 3], [0, 1, 2, 3, 0]),
        (True, [0, 1, 0, 0], [1, 2, 3], [0, 1, 2, 3, 0, 0]),
    ],
)
def test_FFTConvolve(sampling_rate, keep_tail, aux, base, expected):

    with auglib.AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
        transform = auglib.transform.FFTConvolve(aux_buf, keep_tail=keep_tail)
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
            transform(base_buf)
            np.testing.assert_almost_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
                decimal=4,
            )
