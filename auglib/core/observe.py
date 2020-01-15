from typing import Union, Any, Sequence
import random
import warnings

import numpy as np

from .common import Object


def _fit_value(value: Union[int, float], minimum: Union[int, float],
               maximum: Union[int, float]) -> Union[int, float]:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


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


class StrList(Str):
    r"""Iterates over a list of strings.

    .. note:: You can either ``shuffle`` or ``draw``.

    Args:
        strings: list of strings
        shuffle: return strings in random order
        draw: randomly draw the next string

    """
    def __init__(self, strings: Sequence[str], *, shuffle: bool = False,
                 draw: bool = False):
        assert not (draw and shuffle), 'you can not draw and shuffle at the ' \
                                       'same time'

        self.strings = strings
        self.shuffle = shuffle
        self.draw = draw
        self._counter = 0
        self._iter = False

    def __len__(self):
        return len(self.strings)

    def _draw(self) -> str:
        return self.strings[random.randint(0, len(self) - 1)]

    def _next(self) -> str:
        if self.shuffle and self._counter == 0:
            random.shuffle(self.strings)
        file = self.strings[self._counter]
        self._counter += 1
        if self._counter >= len(self):
            self._counter %= len(self)
            self._iter = False
        return file

    def __call__(self) -> str:
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


class BoolUni(Bool):
    r"""Draws a boolean value with a 50/50 chance.
    """
    def __call__(self) -> bool:
        bool_list = [True, False]
        value = bool_list[np.random.randint(0, 2)]
        return value


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
        minimum: minimum value
        maximum: maximum value

    """
    def __init__(self, mean: float, std: float, *,
                 minimum: float = None,
                 maximum: float = None):
        self.mean = mean
        self.std = std
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self) -> float:
        value = _fit_value(np.random.normal(self.mean, self.std),
                           minimum=self.minimum, maximum=self.maximum)
        return value


def observe(x: Union[Any, Observable]) -> Any:
    r"""Convenient function to observe an object.

    Returns ``x()`` if ``x`` is of type :class:`auglib.observe.Observable`,
    otherwise returns ``x``.

    Args:
        x: source

    """
    return x() if isinstance(x, Observable) else x
