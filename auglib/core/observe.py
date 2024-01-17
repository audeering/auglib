import random
import typing

import numpy as np

import audobject


class Base(audobject.Object):
    r"""Interface for observable objects.

    An observable object only reveals its value when it is called.

    """

    def __call__(self) -> typing.Any:  # pragma: no cover
        r"""Observe next value.

        Returns:
            next value

        """
        raise NotImplementedError()


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
        r"""Observe next boolean value.

        Returns:
            boolean

        Examples:
            >>> import auglib
            >>> auglib.seed(1)
            >>> o = Bool(prob_true=0.5)
            >>> o()
            True
            >>> o()
            False

        """
        return np.random.random() <= self.prob_true


class FloatNorm(Base):
    r"""Draw floating point numbers from a normal (Gaussian) distribution.

    Args:
        mean: mean (center) of the distribution
        std: standard deviation (spread) of the distribution
        minimum: lower bound
        maximum: upper bound

    Examples:
        >>> import auglib
        >>> auglib.seed(1)
        >>> o = FloatNorm(mean=0.0, std=1.0, minimum=0.0)
        >>> round(o(), 2)
        1.62
        >>> round(o(), 2)
        0.87

    """

    def __init__(
        self,
        mean: float,
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
        self._gen = _truncnorm

    def __call__(self) -> float:
        r"""Observe next float value.

        Returns:
            float value

        """
        value = self._gen(self.mean, self.std, self.minimum, self.maximum)
        return value


class FloatUni(Base):
    r"""Draw floating point numbers from a uniform distribution.

    Draws from a "half-open" interval defined by ``[low, high)``.

    Args:
        low: low interval (inclusive)
        high: high interval (exclusive)

    Examples:
        >>> import auglib
        >>> auglib.seed(1)
        >>> o = FloatUni(low=0.0, high=1.0)
        >>> round(o(), 2)
        0.42
        >>> round(o(), 2)
        0.72

    """

    def __init__(
        self,
        low: float,
        high: float,
    ):
        self.low = low
        self.high = high

    def __call__(self) -> float:
        r"""Observe next float value.

        Returns:
            float value

        """
        value = np.random.uniform(self.low, self.high)
        return value


class IntUni(Base):
    r"""Draw integers from a uniform distribution.

    Draws from the interval defined by ``[low, high]``.

    Args:
        low: low interval (inclusive)
        high: high interval (inclusive)

    Examples:
        >>> import auglib
        >>> auglib.seed(1)
        >>> o = IntUni(low=0, high=5)
        >>> round(o(), 2)
        5
        >>> round(o(), 2)
        3

    """

    def __init__(
        self,
        low: int,
        high: int,
    ):
        self.low = low
        self.high = high

    def __call__(self) -> int:
        r"""Observe next integer.

        Returns:
            integer

        """
        # self.high + 1 ensures upper bound is included
        # as np.random.randint excludes it
        return np.random.randint(self.low, self.high + 1)


class List(Base):
    r"""Iterate over a list of (observable) objects.

    List can contain objects with any type.
    Objects that implement
    :class:`auglib.observe.Base`
    are automatically observed
    and the observed value is returned.

    Args:
        elements: list of (observables) objects
        shuffle: return elements in random order
        draw: randomly draw the next element

    Raises:
        ValueError: when trying to ``shuffle`` and ``draw`` at the same time

    Examples:
        >>> import auglib
        >>> auglib.seed(1)
        >>> o = List([0, "b", "c"])
        >>> [o() for _ in range(5)]
        [0, 'b', 'c', 0, 'b']
        >>> o = List([0, "b", "c"], shuffle=True)
        >>> [o() for _ in range(5)]
        ['b', 'c', 0, 0, 'b']
        >>> o = List([0, "b", "c"], draw=True)
        >>> [o() for _ in range(5)]
        ['b', 'b', 'b', 'c', 'b']
        >>> o = List([IntUni(0, 5), 99])
        >>> [o() for _ in range(5)]
        [5, 99, 3, 99, 4]

    """

    # import here to avoid circulate import error
    from auglib.core.resolver import ObservableListResolver

    @audobject.init_decorator(
        resolvers={
            "elements": ObservableListResolver,
        }
    )
    def __init__(
        self,
        elements: typing.MutableSequence[typing.Any],
        *,
        shuffle: bool = False,
        draw: bool = False,
    ):
        if draw and shuffle:
            raise ValueError("Cannot draw and shuffle at the same time.")

        self.elements = list(elements)
        self.shuffle = shuffle
        self.draw = draw
        self._counter = 0
        self._iter = False

    def _draw(self) -> typing.Any:
        return self.elements[random.randint(0, len(self) - 1)]

    def _next(self) -> typing.Any:
        if self.shuffle and self._counter == 0:
            random.shuffle(self.elements)
        element = self.elements[self._counter]
        self._counter += 1
        if self._counter >= len(self):
            self._counter %= len(self)
            self._iter = False
        return element

    def __call__(self) -> typing.Any:
        r"""Observe next value from list.

        Returns:
            next value

        """
        if self.draw:
            return observe(self._draw())
        else:
            return observe(self._next())

    def __iter__(self):  # noqa: D105
        self._counter = 0
        self._iter = True
        return self

    def __len__(self):  # noqa: D105
        return len(self.elements)

    def __next__(self):  # noqa: D105
        if not self._iter:
            raise StopIteration()
        return self()


def observe(
    x: typing.Union[typing.Any, Base],
) -> typing.Any:
    r"""Convenient function to observe a value.

    Returns ``x()`` if ``x`` is of type :class:`auglib.observe.Base`,
    otherwise just ``x``.

    Args:
        x: (observable) object

    Returns:
        observed value

    Examples:
        >>> import auglib
        >>> auglib.seed(1)
        >>> observe(99)
        99
        >>> o = IntUni(low=0, high=5)
        >>> observe(o)
        5
        >>> observe(o)
        3

    """
    return x() if isinstance(x, Base) else x


def _truncnorm(
    mu: float,
    sigma: float,
    minimum: float,
    maximum: float,
) -> float:
    r"""Truncated standard normal distribution.

    Alternative to scipy.stats.truncnorm.

    It simply draws from the random distribution
    until the returned value
    is within the given boundary.

    Args:
        mu: mean of distribution
        sigma: standard deviation of distribution
        minimum: lowest value to be returned
        maximum: highest number to be returned

    Returns:
        value drawn from standard deviation within given boundaries

    """
    s = np.random.normal(mu, sigma)
    while s < minimum or s > maximum:
        s = np.random.normal(mu, sigma)
    return s
