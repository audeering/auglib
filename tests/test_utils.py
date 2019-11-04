import pytest

from auglib.utils import to_samples, to_db, from_db


@pytest.mark.parametrize('dur,sr,unit,n',
                         [(1.0, 8000, None, 8000),
                          (8000, 8000, 'samples', 8000),
                          (1.0, 8000, 's', 8000),
                          (1000, 8000, 'ms', 8000),
                          (1 / 60, 8000, 'm', 8000),
                          (1 / 3600, 8000, 'hour', 8000)])
def test_duration(dur, sr, unit, n):
    if unit is None:
        assert to_samples(dur, sr) == n
    else:
        assert to_samples(dur, sr, unit=unit) == n


@pytest.mark.parametrize('gain', [1.0, 10.0])
def test_gain(gain):
    assert from_db(to_db(gain)) == gain
