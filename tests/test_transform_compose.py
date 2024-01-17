import numpy as np
import pytest

import auglib


auglib.seed(0)


@pytest.mark.parametrize("signal", [[1, 1]])
@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize(
    "transforms, expected",
    [
        (
            [
                auglib.transform.AppendValue(1, value=0, unit="samples"),
                auglib.transform.PrependValue(1, value=2, unit="samples"),
            ],
            [2, 1, 1, 0],
        ),
        (
            [
                auglib.transform.PrependValue(1, value=2, unit="samples"),
                auglib.transform.AppendValue(1, value=0, unit="samples"),
            ],
            [2, 1, 1, 0],
        ),
    ],
)
def test_compose(
    signal,
    sampling_rate,
    transforms,
    expected,
):
    signal = np.array(signal)
    expected = np.array(expected, dtype=auglib.core.transform.DTYPE)

    transform = auglib.transform.Compose(transforms)
    np.testing.assert_array_equal(
        transform(signal, sampling_rate),
        expected,
        strict=True,  # ensure same shape and dtype
    )
