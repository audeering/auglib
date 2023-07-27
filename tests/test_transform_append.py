import numpy as np
import pytest

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[1, 1]])
@pytest.mark.parametrize(
    'read_pos_aux, read_dur_aux, unit, aux, expected',
    [
        (0, 0, 'samples', [0, 2], [1, 1, 0, 2]),
        (0, None, 'samples', [0, 2], [1, 1, 0, 2]),
        (1, 0, 'samples', [0, 2], [1, 1, 2]),
        (0, 1, 'samples', [0, 2], [1, 1, 0]),
    ],
)
def test_Append(
        tmpdir,
        sampling_rate,
        base,
        read_pos_aux,
        read_dur_aux,
        unit,
        aux,
        expected,
):
    with auglib.AudioBuffer.from_array(aux, sampling_rate) as aux_buf:

        transform = auglib.transform.Append(
            aux_buf,
            read_pos_aux=read_pos_aux,
            read_dur_aux=read_dur_aux,
            unit=unit,
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
            transform(base_buf)
            np.testing.assert_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
            )
