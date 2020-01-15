import pytest
import numpy as np

from auglib import StrList, FloatNorm, FloatUni, IntUni
from auglib.core.observe import observe, Observable, BoolRand
from auglib.utils import random_seed


@pytest.mark.parametrize('x',
                         [1, 1.0, 'str',
                          IntUni(0, 10),
                          FloatUni(0.0, 1.0),
                          FloatNorm(0.0, 1.0),
                          StrList(['a', 'b', 'c'], draw=True)])
def test_observe(x):
    if isinstance(x, Observable):
        random_seed(1)
        tmp = x()
        random_seed(1)
        assert tmp == observe(x)
    else:
        assert x == observe(x)


@pytest.mark.parametrize('n', [1000])
def test_BoolRand(n):
    x = BoolRand()
    draws = []
    for _ in range(n):
        draws += [x()]
    assert (True in draws) and (False in draws)


@pytest.mark.parametrize('n,low,high',
                         ([1000, 0, 100], ))
def test_IntUni(n, low, high):
    x = IntUni(low, high)
    for _ in range(n):
        assert low <= x() < high


@pytest.mark.parametrize('n,low,high',
                         ([1000, 0.0, 1.0], ))
def test_FloatUni(n, low, high):
    x = FloatUni(low, high)
    for _ in range(n):
        assert low <= x() < high


@pytest.mark.parametrize('n,mean,std,minimum,maximum',
                         ([10000, 0.0, 1.0, -0.1, 0.1], ))
def test_FloatNorm(n, mean, std, minimum, maximum):
    x = FloatNorm(mean, std)
    y = np.array([x() for _ in range(n)])
    np.testing.assert_almost_equal(mean, y.mean(), decimal=1)
    np.testing.assert_almost_equal(std, y.std(), decimal=1)
    x = FloatNorm(mean, std, minimum=minimum, maximum=maximum)
    y = np.array([x() for _ in range(n)])
    assert min(y) >= minimum
    assert max(y) <= maximum


@pytest.mark.parametrize('n,strings',
                         ([100, ['a', 'b', 'c']], ))
def test_StrList(n, strings):
    x = StrList(strings)
    for s in strings:
        assert x() == s
    x = StrList(strings.copy(), shuffle=True)
    assert strings == sorted([s for s in x])
    x = StrList(strings, draw=True)
    for _ in range(n):
        assert x() in strings
