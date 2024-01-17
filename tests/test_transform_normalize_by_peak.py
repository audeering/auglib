import numpy as np
import pytest

import audmath
import audobject

import auglib


@pytest.mark.parametrize(
    "signal, peak_db, expected_peak",
    [
        (np.linspace(-0.5, 0.5, num=10), 0, 1),
        (np.linspace(-0.5, 0.5, num=10), -3, audmath.inverse_db(-3)),
        (np.linspace(-0.5, 0.5, num=10), 3, audmath.inverse_db(3)),
        (np.zeros((1, 10)), 0, 0),
        (np.zeros((1, 10)), -3, 0),
        (np.zeros((1, 10)), 3, 0),
    ],
)
def test_Normalize(signal, peak_db, expected_peak):
    transform = auglib.transform.NormalizeByPeak(peak_db=peak_db)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )
    augmented_signal = transform(signal)
    assert augmented_signal.dtype == auglib.core.transform.DTYPE
    assert np.allclose(
        np.abs(augmented_signal).max(),
        expected_peak,
        atol=1e-6,
    )
