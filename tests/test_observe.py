import numpy as np
import pytest

import auglib


@pytest.mark.parametrize(
    "x",
    [
        1,
        1.0,
        "str",
        auglib.observe.Bool(),
        auglib.observe.IntUni(0, 10),
        auglib.observe.FloatUni(0.0, 1.0),
        auglib.observe.FloatNorm(0.0, 1.0),
        auglib.observe.List(["a", "b", "c"], draw=True),
    ],
)
def test_observe(x):
    if isinstance(x, auglib.observe.Base):
        auglib.seed(0)
        tmp = x()
        auglib.seed(0)
        assert tmp == auglib.observe.observe(x)
    else:
        assert x == auglib.observe.observe(x)


@pytest.mark.parametrize("n", [1000])
def test_Bool(n):
    x = auglib.observe.Bool()
    draws = []
    for _ in range(n):
        draws += [x()]
    assert (True in draws) and (False in draws)


@pytest.mark.parametrize("n,low,high", ([1000, 0, 100],))
def test_IntUni(n, low, high):
    x = auglib.observe.IntUni(low, high)
    for _ in range(n):
        assert low <= x() <= high


@pytest.mark.parametrize("n,low,high", ([1000, 0.0, 1.0],))
def test_FloatUni(n, low, high):
    x = auglib.observe.FloatUni(low, high)
    for _ in range(n):
        assert low <= x() < high


@pytest.mark.parametrize("n,mean,std,minimum,maximum", ([10000, 0.0, 1.0, -0.1, 0.1],))
def test_FloatNorm(n, mean, std, minimum, maximum):
    x = auglib.observe.FloatNorm(mean, std)
    y = np.array([x() for _ in range(n)])
    np.testing.assert_almost_equal(mean, y.mean(), decimal=1)
    np.testing.assert_almost_equal(std, y.std(), decimal=1)
    x = auglib.observe.FloatNorm(mean, std, minimum=minimum, maximum=maximum)
    y = np.array([x() for _ in range(n)])
    assert min(y) >= minimum
    assert max(y) <= maximum


@pytest.mark.parametrize(
    "n,elements",
    (
        [100, ["a", "b", "c"]],
        [100, [1, 2, 3]],
        [100, [1.0, 2.0, 3.0]],
    ),
)
def test_List(n, elements):
    x = auglib.observe.List(elements)
    for s in elements:
        assert x() == s

    x = auglib.observe.List(elements.copy(), shuffle=True)
    assert elements == sorted([s for s in x])

    x = auglib.observe.List(elements, draw=True)
    for _ in range(n):
        assert x() in elements

    with pytest.raises(ValueError):
        auglib.observe.List(elements.copy(), shuffle=True, draw=True)
