import numpy as np
import pytest

import audobject

import auglib


def func_double(x, sr):
    import numpy as np

    return np.tile(x, 2)


def func_empty(x, sr):
    import numpy as np

    return np.array([])


def func_plus_c(x, sr, c):
    return x + c


def func_times_2(x, sr):
    return 2 * x


@pytest.mark.parametrize(
    "base, function, function_args, expected",
    [
        (
            # add 1 to signal
            np.zeros(20),
            func_plus_c,
            {"c": auglib.observe.IntUni(1, 1)},
            np.ones(20),
        ),
        (
            np.zeros(1),
            func_plus_c,
            {"c": 1},
            np.ones(1),
        ),
        (
            np.zeros((1, 1)),
            func_plus_c,
            {"c": 1},
            np.ones((1, 1)),
        ),
        (
            # halve signal size
            np.ones(20),
            lambda x, sr: x[:, ::2],
            None,
            np.ones(10),
        ),
        (
            # multiple by 2
            np.ones(10),
            func_times_2,
            None,
            np.ones(10) * 2,
        ),
        (
            # double signal size
            np.ones(10),
            func_double,
            None,
            np.ones(20),
        ),
        (
            # return empty signal
            np.ones(20),
            func_empty,
            None,
            np.array([]),
        ),
    ],
)
def test_Function(base, function, function_args, expected):
    expected = expected.astype(auglib.core.transform.DTYPE)

    transform = auglib.transform.Function(
        function,
        function_args,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )
    np.testing.assert_array_equal(
        transform(base),
        expected,
        strict=True,  # ensure same shape and dtype
    )
