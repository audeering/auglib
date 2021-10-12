import random
import typing

import numpy as np
import scipy.stats

import audobject


# abstract

class Base(audobject.Object):
    r"""An observable object.

    An observable object only reveals its value when it is called.

    Example:
        >>> s = StrList(['a', 'b', 'c'])
        >>> s()
        'a'
        >>> s()
        'b'
        >>> s()
        'c'

    """
    def __call__(self) -> typing.Any:
        raise (NotImplementedError())


# implementations


class Bool(Base):
    r"""Draw booleans with a given probability for True.

    Args:
        prob_true: probability for True values to be drawn

    """
    def __init__(
            self,
            prob_true: float = 0.5,
    ):
        self.prob_true = prob_true

    def __call__(self) -> bool:
        return np.random.random() <= self.prob_true


class FloatNorm(Base):
    r"""Draw floating point numbers from a normal (Gaussian) distribution.

    Args:
        mean: mean (center) of the distribution
        std: standard deviation (spread) of the distribution
        minimum: lower bound
        maximum: upper bound

    """
    def __init__(
            self, mean: float,
            std: float,
            *,
            minimum: float = None,
            maximum: float = None,
    ):
        minimum = minimum if minimum is not None else -np.inf
        maximum = maximum if maximum is not None else np.inf
        self.mean = mean
        self.std = std
        self.minimum = minimum
        self.maximum = maximum
        self._gen = scipy.stats.truncnorm(
            (minimum - mean) / std,
            (maximum - mean) / std,
            loc=mean,
            scale=std,
        )

    def __call__(self) -> float:
        value = self._gen.rvs()
        return value


class FloatUni(Base):
    r"""Draw floating point numbers from a uniform distribution.

    Args:
        low: lower bound of the support interval (inclusive)
        high: upper bound of the support interval (exclusive)

    """
    def __init__(
            self,
            low: float,
            high: float,
    ):
        self.low = low
        self.high = high

    def __call__(self) -> float:
        value = np.random.uniform(self.low, self.high)
        return value


class IntUni(Base):
    r"""Draw integers from a uniform distribution.

    Args:
        low: low interval (inclusive)
        high: high interval (exclusive)

    """
    def __init__(
            self,
            low: int,
            high: int,
    ):
        self.low = low
        self.high = high

    def __call__(self) -> int:
        return np.random.randint(self.low, self.high)


# lists


class List(Base):
    r"""Iterate over a list of observable objects.

    Args:
        elements: list of observables
        shuffle: return elements in random order
        draw: randomly draw the next element

    Raises:
        ValueError: when trying to ``shuffle`` and ``draw`` at the same time

    """
    def __init__(
            self,
            elements: typing.MutableSequence[typing.Union[int, float, str]],
            *,
            shuffle: bool = False,
            draw: bool = False,
    ):

        if draw and shuffle:
            raise ValueError("Cannot draw and shuffle at the same time.")

        self.elements = elements
        self.shuffle = shuffle
        self.draw = draw
        self._counter = 0
        self._iter = False

    def __len__(self):
        return len(self.elements)

    def _draw(self) -> typing.Union[int, float, str]:
        return self.elements[random.randint(0, len(self) - 1)]

    def _next(self) -> typing.Union[int, float, str]:
        if self.shuffle and self._counter == 0:
            random.shuffle(self.elements)
        element = self.elements[self._counter]
        self._counter += 1
        if self._counter >= len(self):
            self._counter %= len(self)
            self._iter = False
        return element

    def __call__(self) -> typing.Union[int, float, str]:
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


class FloatList(List):
    r"""Iterate over a list of floats.

    Args:
        elements: list of floats
        shuffle: return elements in random order
        draw: randomly draw the next element

    Raises:
        ValueError: when trying to ``shuffle`` and ``draw`` at the same time

    """
    def __init__(
            self,
            elements: typing.MutableSequence[float],
            *,
            shuffle: bool = False,
            draw: bool = False,
    ):
        super().__init__(elements, shuffle=shuffle, draw=draw)


class IntList(List):
    r"""Iterate over a list of integers.

    Args:
        elements: list of integers
        shuffle: return elements in random order
        draw: randomly draw the next element

    Raises:
        ValueError: when trying to ``shuffle`` and ``draw`` at the same time

    """
    def __init__(
            self,
            elements: typing.MutableSequence[int],
            *,
            shuffle: bool = False,
            draw: bool = False,
    ):
        super().__init__(elements, shuffle=shuffle, draw=draw)


class StrList(List):
    r"""Iterate over a list of strings.

    Args:
        elements: list of strings
        shuffle: return elements in random order
        draw: randomly draw the next element

    Raises:
        ValueError: when trying to ``shuffle`` and ``draw`` at the same time

    """
    def __init__(
            self,
            elements: typing.MutableSequence[str],
            *,
            shuffle: bool = False,
            draw: bool = False,
    ):
        super().__init__(elements, shuffle=shuffle, draw=draw)


def observe(
        x: typing.Union[typing.Any, Base],
) -> typing.Any:
    r"""Convenient function to observe an object.

    Returns ``x()`` if ``x`` is of type :class:`auglib.observe.Base`,
    otherwise returns ``x``.

    Args:
        x: source

    """
    return x() if isinstance(x, Base) else x
