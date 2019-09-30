from typing import Union

import humanfriendly as hf
import numpy as np


def gain_to_db(gain: float) -> float:
    r"""Convert gain to decibels (dB).

    Args:
        gain: input gain

    """
    assert gain > 0
    gain_db = 20 * np.log10(gain)
    return gain_db


def db_to_gain(gain_db: float) -> float:
    r"""Convert decibels (dB) to gain.

    Args:
        gain_db: input gain in decibels

    """
    gain = pow(10.0, gain_db / 20.0)
    return gain


def dur_to_samples(dur: float,
                   sampling_rate: int,
                   *,
                   unit: str = 'seconds') -> int:
    r"""Express duration in samples.

    Examples (``sample_rate==8000``):

    =======  =========  ===================
    dur      unit       result (in samples)
    =======  =========  ===================
    1.0                 8000
    8000     'samples'  8000
    2/3600   'hour'     16000
    500      'ms'       4000
    0.5      's'        4000
    =======  =========  ===================

    Args:
        dur: duration (see description)
        sampling_rate: sampling rate in Hz
        unit: literal specifying the format

    """

    unit = unit.strip()
    if unit == 'samples':
        return int(dur)
    else:
        return int(hf.parse_timespan(str(dur) + unit) * sampling_rate)
