import pytest
import numpy as np

from auglib.utils import dur2samples, gain2db


@pytest.mark.parametrize('dur,sr,unit,n',
                         [(1.0, 8000, None, 8000),
                          (8000, 8000, 'samples', 8000),
                          (1.0, 8000, 's', 8000),
                          (1000, 8000, 'ms', 8000),
                          (1 / 60, 8000, 'm', 8000),
                          (1 / 3600, 8000, 'hour', 8000)])
def test_dur2samples(dur, sr, unit, n):
    if unit is None:
        assert dur2samples(dur, sr) == n
    else:
        assert dur2samples(dur, sr, unit=unit) == n


@pytest.mark.parametrize('gain', [1.0, 10.0])
def test_gain2db(gain):
    assert gain2db(gain) == 20 * np.log10(gain)
