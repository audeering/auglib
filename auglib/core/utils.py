import random
import typing

import numpy as np
import pandas as pd

import audeer

from auglib.core.api import lib
from auglib.core import observe
from auglib.core import time


def from_db(x_db: typing.Union[float, observe.Base]) -> float:
    r"""Convert decibels (dB) to gain.

    Args:
        x_db: input gain in decibels

    Returns:
        input gain

    Example:
        >>> from_db(-3)
        0.7079457843841379

    """
    x_db = observe.observe(x_db)
    x = pow(10.0, x_db / 20.0)
    return x


@audeer.deprecated(removal_version='1.0.0', alternative='auglib.seed')
def random_seed(seed: int):  # pragma: no cover
    r"""(Re-)initialize random generator.

    The random generator is shared among all audio classes.

    Args:
        seed: seed number (0 for random initialization)

    """
    random.seed(None if seed == 0 else seed)
    np.random.seed(None if seed == 0 else seed)
    lib.auglib_random_seed(seed)


def to_db(x: typing.Union[float, observe.Base]) -> float:
    r"""Convert gain to decibels (dB).

    Args:
        x: input gain

    Returns:
        input gain in dB

    Example:
        >>> to_db(2)
        6.020599913279624

    """
    x = observe.observe(x)
    assert x > 0, 'cannot convert gain {} to decibels'.format(x)
    x_db = 20 * np.log10(x)
    return x_db


def to_samples(
        value: typing.Union[int, float, observe.Base, time.Time],
        *,
        sampling_rate: int = None,
        length: int = None,
        unit: str = 'seconds',
        allow_negative: bool = False,
) -> int:
    r"""Express timestamp or timespan in samples.

    If ``unit`` is set to ``'samples'``,
    no argument must be given.
    In case of ``'relative'``,
    a value for ``length`` has to be provided.
    In any other case,
    a value for ``sampling_rate`` is required
    and ``unit`` must be supported by
    :func:`pandas.to_timedelta`.

    Examples for a ``sampling_rate`` of 8000,
    highlighting the influence of ``unit``:

    =======  ===========  =======  ====================
    value    unit         length   result (in samples)
    =======  ===========  =======  ====================
    1.0      'seconds'             8000
    8000     'samples'             8000
    2/3600   'hour'                16000
    500      'ms'                  4000
    0.5      's'                   4000
    0.25     'relative'   8000     2000
    =======  ===========  =======  ====================

    Args:
        value: timestamp or timespan
        sampling_rate: sampling rate in Hz
        length: reference point if unit is ``relative`` (in number of samples)
        unit: literal specifying the format
            (ignored if ``value`` has type :class:`auglib.Time`)
        allow_negative: allow negative values

    Returns:
        number of samples

    Raises:
        ValueError: if ``allow_negative`` is ``False``
            and computed value is negative
        ValueError: if ``unit`` is not supported
        ValueError: if ``length`` is not provided,
            but ``unit`` is ``'samples'``
        ValueError: if  ``sampling_rate`` is not provided,
            but ``unit`` is not ``'samples'`` or ``'relative'``

    Example:
        >>> to_samples(0.5, sampling_rate=10)
        5
        >>> to_samples(0.5, length=20, unit='relative')
        10
        >>> to_samples(time.Time(1500, unit='ms'), sampling_rate=10)
        15

    """
    if not isinstance(value, time.Time):
        value = time.Time(value, unit)
    return value(
        sampling_rate=sampling_rate,
        length=length,
        allow_negative=allow_negative,
    )
