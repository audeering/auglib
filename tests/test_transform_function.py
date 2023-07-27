import re

import numpy as np
import pytest

import audobject

import auglib


def test_Function():

    def func_double(x, sr):
        import numpy as np
        return np.tile(x, 2)

    def func_plus_c(x, sr, c):
        return x + c

    def func_times_2(x, sr):  # inplace
        x *= 2

    with auglib.AudioBuffer(20, 16000, unit='samples') as buffer:

        np.testing.assert_equal(
            buffer._data,
            np.zeros(20, dtype=np.float32),
        )

        # add 1 to buffer
        transform = auglib.transform.Function(
            func_plus_c,
            {'c': auglib.observe.IntUni(1, 1)},
        )
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(20, dtype=np.float32),
        )

        # halve buffer size
        transform = auglib.transform.Function(lambda x, sr: x[:, ::2])
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(10, dtype=np.float32),
        )

        # multiple by 2
        transform = auglib.transform.Function(func_times_2)
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(10, dtype=np.float32) * 2,
        )

        # double buffer size
        transform = auglib.transform.Function(func_double)
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )
        transform(buffer)
        np.testing.assert_equal(
            buffer._data,
            np.ones(20, dtype=np.float32) * 2,
        )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'base, function, error, error_msg',
    [
        (
            [0, 0],
            lambda sig, sr: np.array([]),
            RuntimeError,
            (
                'Buffers must be non-empty. '
                'Yours is empty '
                'after applying the following transform: '
                "'$auglib.core.transform.Function"
            ),
        ),
    ],
)
def test_Function_errors(
        sampling_rate,
        base,
        function,
        error,
        error_msg,
):

    with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
        with pytest.raises(error, match=re.escape(error_msg)):
            auglib.transform.Function(function)(base_buf)
