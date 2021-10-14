from typing import Union
import random
import os

import humanfriendly as hf
import numpy as np

import audeer

from auglib.core.api import lib
from auglib.core import observe


def assert_non_negative_number(value: Union[int, float]):
    if value < 0:
        raise ValueError('A variable that is supposed to be non-negative was '
                         'found negative.')


def from_db(x_db: Union[float, observe.Base]) -> float:
    r"""Convert decibels (dB) to gain.

    Args:
        x_db: input gain in decibels

    """
    x_db = observe.observe(x_db)
    x = pow(10.0, x_db / 20.0)
    return x


def random_seed(seed: int = 0):
    r"""(Re-)initialize random generator..

    .. note:: Controls a random generator that is shared among all audio
        classes.

    Args:
        seed: seed number (0 for random initialization)

    """
    random.seed(None if seed == 0 else seed)
    np.random.seed(None if seed == 0 else seed)
    lib.auglib_random_seed(seed)


def to_db(x: Union[float, observe.Base]) -> float:
    r"""Convert gain to decibels (dB).

    Args:
        x: input gain

    """
    x = observe.observe(x)
    assert x > 0, 'cannot convert gain {} to decibels'.format(x)
    x_db = 20 * np.log10(x)
    return x_db


def to_samples(value: Union[int, float, observe.Base],
               sampling_rate: int,
               *,
               length: int = 0,
               unit: str = 'seconds',
               allow_negative: bool = False) -> int:
    r"""Express duration in samples.

    Examples (``sample_rate==8000``):

    =======  ===========  =======  ====================
    value    unit         length   result (in samples)
    =======  ===========  =======  ====================
    1.0                            8000
    8000     'samples'             8000
    2/3600   'hour'                16000
    500      'ms'                  4000
    0.5      's'                   4000
    0.25     'relative'   8000     2000
    =======  ===========  =======  ====================

    Args:
        value: duration or portion (see description)
        sampling_rate: sampling rate in Hz
        length: reference point if unit is ``relative`` (in number of samples)
        unit: literal specifying the format
        allow_negative: allow negative values

    Raises:
        ValueError: if ``allow_negative`` is False and computed value is
            negative

    """
    value = observe.observe(value)
    unit = unit.strip()
    if unit == 'samples':
        num_samples = int(value)
    elif unit == 'relative':
        if not 0.0 <= value <= 1.0:
            raise ValueError('relative value {} not in range ['
                             '0...1]'.format(value))
        num_samples = int(length * value)
    else:
        try:
            num_samples = int(
                hf.parse_timespan(str(value) + unit) * sampling_rate
            )
        except hf.InvalidTimespan as ex:
            raise ValueError(str(ex))
    if not allow_negative:
        assert_non_negative_number(num_samples)
    return num_samples
