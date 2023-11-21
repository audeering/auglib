import re

import numpy as np
import pytest

import auglib


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'base, aux, read_pos_aux, read_dur_aux, unit, expected',
    [
        ([1, 1], [0, 2], 0, 0, 'samples', [1, 1, 0, 2]),
        ([1, 1], [0, 2], 0, None, 'samples', [1, 1, 0, 2]),
        ([1, 1], [0, 2], 1, 0, 'samples', [1, 1, 2]),
        ([1, 1], [0, 2], 0, 1, 'samples', [1, 1, 0]),
        ([1, 1], [], 0, None, 'samples', [1, 1]),
        ([1, 1], [[0, 2]], 0, 0, 'samples', [1, 1, 0, 2]),
        ([1, 1], [[0, 2]], 0, None, 'samples', [1, 1, 0, 2]),
        ([1, 1], [[0, 2]], 1, 0, 'samples', [1, 1, 2]),
        ([1, 1], [[0, 2]], 0, 1, 'samples', [1, 1, 0]),
        ([1, 1], [[]], 0, None, 'samples', [1, 1]),
        ([[1, 1]], [0, 2], 0, 0, 'samples', [[1, 1, 0, 2]]),
        ([[1, 1]], [0, 2], 0, None, 'samples', [[1, 1, 0, 2]]),
        ([[1, 1]], [0, 2], 1, 0, 'samples', [[1, 1, 2]]),
        ([[1, 1]], [0, 2], 0, 1, 'samples', [[1, 1, 0]]),
        ([[1, 1]], [], 0, None, 'samples', [[1, 1]]),
        ([[1, 1]], [[0, 2]], 0, 0, 'samples', [[1, 1, 0, 2]]),
        ([[1, 1]], [[0, 2]], 0, None, 'samples', [[1, 1, 0, 2]]),
        ([[1, 1]], [[0, 2]], 1, 0, 'samples', [[1, 1, 2]]),
        ([[1, 1]], [[0, 2]], 0, 1, 'samples', [[1, 1, 0]]),
        ([[1, 1]], [[]], 0, None, 'samples', [[1, 1]]),
    ],
)
def test_append(
        tmpdir,
        sampling_rate,
        base,
        read_pos_aux,
        read_dur_aux,
        unit,
        aux,
        expected,
):
    base = np.array(base)
    aux = np.array(aux)
    expected = np.array(expected, dtype=auglib.core.transform.DTYPE)

    transform = auglib.transform.Append(
        aux,
        read_pos_aux=read_pos_aux,
        read_dur_aux=read_dur_aux,
        sampling_rate=sampling_rate,
        unit=unit,
    )
    np.testing.assert_array_equal(
        transform(base),
        expected,
        strict=True,  # ensure same shape and dtype
    )


def test_append_error():
    transform = auglib.transform.Append(np.array([0, 1]))
    error_msg = (
        "Cannot serialize an instance of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and pass filename."
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        transform.to_yaml_s(include_version=False)
