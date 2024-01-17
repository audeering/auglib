import numpy as np
import pytest

import auglib


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("signal", [np.array([1, 1])])
@pytest.mark.parametrize(
    "read_pos_aux, read_dur_aux, unit, aux, expected",
    [
        (
            0,
            None,
            "samples",
            np.array([0, 2]),
            np.array([0, 2, 1, 1]),
        ),
        (
            0,
            0,
            "samples",
            np.array([0, 2]),
            np.array([0, 2, 1, 1]),
        ),
        (
            1,
            None,
            "samples",
            np.array([0, 2]),
            np.array([2, 1, 1]),
        ),
        (
            0,
            1,
            "samples",
            np.array([0, 2]),
            np.array([0, 1, 1]),
        ),
        (
            0,
            None,
            "relative",
            np.array([0, 2]),
            np.array([0, 2, 1, 1]),
        ),
        (
            0.5,
            None,
            "relative",
            np.array([0, 2]),
            np.array([2, 1, 1]),
        ),
        (
            0,
            0.5,
            "relative",
            np.array([0, 2]),
            np.array([0, 1, 1]),
        ),
        (
            0,
            1.5,
            "relative",
            np.array([0, 2]),
            np.array([0, 2, 1, 1]),
        ),
    ],
)
def test_Prepend(
    tmpdir,
    sampling_rate,
    signal,
    read_pos_aux,
    read_dur_aux,
    unit,
    aux,
    expected,
):
    expected = expected.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.Prepend(
        aux,
        read_pos_aux=read_pos_aux,
        read_dur_aux=read_dur_aux,
        unit=unit,
    )
    np.testing.assert_array_equal(
        transform(signal, sampling_rate),
        expected,
        strict=True,
    )


def test_prepend_error():
    transform = auglib.transform.Prepend(np.array([0, 1]))
    error_msg = (
        "Cannot serialize an instance "
        "of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and pass filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)
