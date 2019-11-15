import numpy as np
import pytest
from auglib.transform import NormalizeByPeak
from auglib.interface import NumpyTransform


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_normalize_with_interface(n, sr):

    x = np.linspace(-0.5, 0.5, num=n)
    t = NumpyTransform(NormalizeByPeak(), sr)
    assert np.abs(t(x)).max() == 1.0
