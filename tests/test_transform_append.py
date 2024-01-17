import re

import numpy as np
import pytest

import auglib


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize(
    "signal, aux, read_pos_aux, read_dur_aux, unit, expected",
    [
        ([1, 1], [0, 2], 0, 0, "samples", [1, 1, 0, 2]),
        ([1, 1], [0, 2], 0, None, "samples", [1, 1, 0, 2]),
        ([1, 1], [0, 2], 1, 0, "samples", [1, 1, 2]),
        ([1, 1], [0, 2], 0, 1, "samples", [1, 1, 0]),
        ([1, 1], [], 0, None, "samples", [1, 1]),
        ([1, 1], [[0, 2]], 0, 0, "samples", [1, 1, 0, 2]),
        ([1, 1], [[0, 2]], 0, None, "samples", [1, 1, 0, 2]),
        ([1, 1], [[0, 2]], 1, 0, "samples", [1, 1, 2]),
        ([1, 1], [[0, 2]], 0, 1, "samples", [1, 1, 0]),
        ([1, 1], [[]], 0, None, "samples", [1, 1]),
        ([[1, 1]], [0, 2], 0, 0, "samples", [[1, 1, 0, 2]]),
        ([[1, 1]], [0, 2], 0, None, "samples", [[1, 1, 0, 2]]),
        ([[1, 1]], [0, 2], 1, 0, "samples", [[1, 1, 2]]),
        ([[1, 1]], [0, 2], 0, 1, "samples", [[1, 1, 0]]),
        ([[1, 1]], [], 0, None, "samples", [[1, 1]]),
        ([[1, 1]], [[0, 2]], 0, 0, "samples", [[1, 1, 0, 2]]),
        ([[1, 1]], [[0, 2]], 0, None, "samples", [[1, 1, 0, 2]]),
        ([[1, 1]], [[0, 2]], 1, 0, "samples", [[1, 1, 2]]),
        ([[1, 1]], [[0, 2]], 0, 1, "samples", [[1, 1, 0]]),
        ([[1, 1]], [[]], 0, None, "samples", [[1, 1]]),
        ([[1, 1]], [[0, 2]], 0.5, None, "relative", [[1, 1, 2]]),
        ([[1, 1]], [[0, 2]], 0.5, 0.5, "relative", [[1, 1, 2]]),
        ([[1, 1]], [[0, 2]], 0.5, 1, "relative", [[1, 1, 2]]),
        ([[1, 1]], [[0, 2]], 0, 0.5, "relative", [[1, 1, 0]]),
    ],
)
def test_append(
    tmpdir,
    sampling_rate,
    signal,
    read_pos_aux,
    read_dur_aux,
    unit,
    aux,
    expected,
):
    signal = np.array(signal)
    aux = np.array(aux)
    expected = np.array(expected, dtype=auglib.core.transform.DTYPE)

    transform = auglib.transform.Append(
        aux,
        read_pos_aux=read_pos_aux,
        read_dur_aux=read_dur_aux,
        unit=unit,
    )
    np.testing.assert_array_equal(
        transform(signal, sampling_rate),
        expected,
        strict=True,  # ensure same shape and dtype
    )


@pytest.mark.parametrize(
    "signal, sampling_rate, read_pos_aux, unit, aux, serialize, "
    "expected_error, expected_error_message",
    [
        (
            np.ones((1, 8000)),
            None,
            0.1,
            "seconds",
            np.zeros((1, 2000)),
            False,
            ValueError,
            (
                "Unit is set to 'seconds', "
                "but no value is provided for 'sampling_rate'."
            ),
        ),
        (
            np.ones((1, 4)),
            None,
            None,
            "samples",
            np.array([0, 1]),
            True,
            ValueError,
            (
                "Cannot serialize an instance of <class 'numpy.ndarray'>. "
                "As a workaround, save signal to disk and pass filename."
            ),
        ),
    ],
)
def test_append_error(
    signal,
    sampling_rate,
    read_pos_aux,
    unit,
    aux,
    serialize,
    expected_error,
    expected_error_message,
):
    transform = auglib.transform.Append(
        aux,
        read_pos_aux=read_pos_aux,
        unit=unit,
    )
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        if serialize:
            transform.to_yaml_s(include_version=False)
        else:
            transform(signal, sampling_rate)
