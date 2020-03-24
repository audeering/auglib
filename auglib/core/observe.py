from typing import Union, Any, Sequence
import random

import numpy as np
import scipy.stats

from .common import Object


class Observable(Object):
    r"""An observable object.

    An observable object only reveals its value when it is called.

    Example:
        >>> from auglib import StrList
        >>> s = StrList(['a', 'b', 'c'], shuffle=True)
        >>> s()
        b
        >>> s()
        a
        >>> s()
        c

    """
    def __call__(self) -> Any:
        raise (NotImplementedError())


class Str(Observable):
    r"""An observable string object"""
    def __call__(self) -> str:
        raise (NotImplementedError())


class Number(Observable):
    r"""An observable integer or floating point number."""
    def __call__(self) -> Union[int, float]:
        raise(NotImplementedError())


class Bool(Observable):
    r"""An observable boolean."""
    def __call__(self) -> bool:
        raise(NotImplementedError())


class Int(Number):
    r"""An observable integer."""
    def __call__(self) -> int:
        raise(NotImplementedError())


class Float(Number):
    r"""An observable floating point number."""
    def __call__(self) -> float:
        raise(NotImplementedError())


class ObservableList(Observable):
    r"""Iterates over a list of observable objects.

    .. note:: You can either ``shuffle`` or ``draw``.

    Args:
        elements: list of observables
        shuffle: return elements in random order
        draw: randomly draw the next element

    """
    def __init__(self, elements: Sequence[Union[int, float, str]], *,
                 shuffle: bool = False, draw: bool = False):
        assert not (draw and shuffle), 'you can not draw and shuffle at the ' \
                                       'same time'

        self.elements = elements
        self.shuffle = shuffle
        self.draw = draw
        self._counter = 0
        self._iter = False

    def __len__(self):
        return len(self.elements)

    def _draw(self) -> Union[int, float, str]:
        return self.elements[random.randint(0, len(self) - 1)]

    def _next(self) -> Union[int, float, str]:
        if self.shuffle and self._counter == 0:
            random.shuffle(self.elements)
        element = self.elements[self._counter]
        self._counter += 1
        if self._counter >= len(self):
            self._counter %= len(self)
            self._iter = False
        return element

    def __call__(self) -> Union[int, float, str]:
        if self.draw:
            return self._draw()
        else:
            return self._next()

    def __iter__(self):
        self._counter = 0
        self._iter = True
        return self

    def __next__(self):
        if not self._iter:
            raise StopIteration()
        return self()


class StrList(ObservableList, Str):
    r"""Iterates over a list of strings.

    .. note:: You can either ``shuffle`` or ``draw``.

    Args:
        elements: list of strings
        shuffle: return elements in random order
        draw: randomly draw the next element
    """
    def __init__(self, elements: Sequence[str], *, shuffle: bool = False,
                 draw: bool = False):
        super().__init__(elements, shuffle=shuffle, draw=draw)


class IntList(ObservableList, Int):
    r"""Iterates over a list of integers.

    .. note:: You can either ``shuffle`` or ``draw``.

    Args:
        elements: list of integers
        shuffle: return elements in random order
        draw: randomly draw the next element
    """
    def __init__(self, elements: Sequence[int], *, shuffle: bool = False,
                 draw: bool = False):
        super().__init__(elements, shuffle=shuffle, draw=draw)


class FloatList(ObservableList, Float):
    r"""Iterates over a list of floats.

    .. note:: You can either ``shuffle`` or ``draw``.

    Args:
        elements: list of floats
        shuffle: return elements in random order
        draw: randomly draw the next element
    """
    def __init__(self, elements: Sequence[float], *, shuffle: bool = False,
                 draw: bool = False):
        super().__init__(elements, shuffle=shuffle, draw=draw)


class BoolRand(Bool):
    r"""Draws a boolean value with a given probability for True.

    Args:
        prob_true: probability for True values to be drawn (the probability for
            False values is simply 1.0 - prob_true)
    """
    def __init__(self, prob_true: float = 0.5):
        self.prob_true = prob_true

    def __call__(self) -> bool:
        return np.random.random() <= self.prob_true


class IntUni(Int):
    r"""Draws integers from a uniform distribution.

    Args:
        low: low interval (inclusive)
        high: high interval (exclusive)

    """
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __call__(self) -> int:
        return np.random.randint(self.low, self.high)


class FloatUni(Float):
    r"""Draws floating point numbers from a uniform distribution.

    Args:
        low: lower bound of the support interval (inclusive)
        high: upper bound of the support interval (exclusive)

    """
    def __init__(self, low: float, high: float, *, as_db=False):
        self.low = low
        self.high = high

    def __call__(self) -> float:
        value = np.random.uniform(self.low, self.high)
        return value


class FloatNorm(Float):
    r"""Draws floating point numbers from a normal (Gaussian) distribution.

    Args:
        mean: mean (center) of the distribution
        std: standard deviation (spread) of the distribution
        minimum: lower bound
        maximum: upper bound

    """
    def __init__(self, mean: float, std: float, *,
                 minimum: float = None, maximum: float = None):
        minimum = minimum if minimum is not None else -np.inf
        maximum = maximum if maximum is not None else np.inf
        self._gen = scipy.stats.truncnorm((minimum - mean) / std,
                                          (maximum - mean) / std,
                                          loc=mean,
                                          scale=std)

    def __call__(self) -> float:
        value = self._gen.rvs()
        return value


def observe(x: Union[Any, Observable]) -> Any:
    r"""Convenient function to observe an object.

    Returns ``x()`` if ``x`` is of type :class:`auglib.observe.Observable`,
    otherwise returns ``x``.

    Args:
        x: source

    """
    return x() if isinstance(x, Observable) else x
