from typing import Union, Iterator
import random
import os
import fnmatch
import re

import humanfriendly as hf
import numpy as np

from .api import lib
from .observe import observe, Number, Str


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


def to_db(x: Union[float, Number]) -> float:
    r"""Convert gain to decibels (dB).

    Args:
        x: input gain

    """
    x = observe(x)
    assert x > 0, 'cannot convert gain {} to decibels'.format(x)
    x_db = 20 * np.log10(x)
    return x_db


def from_db(x_db: Union[float, Number]) -> float:
    r"""Convert decibels (dB) to gain.

    Args:
        x_db: input gain in decibels

    """
    x_db = observe(x_db)
    x = pow(10.0, x_db / 20.0)
    return x


def to_samples(value: Union[int, float, Number],
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
    value = observe(value)
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


def _scan_files(root: str,
                sub_dir: str = '',
                recursive: bool = False,
                max_depth: int = None) -> (str, str):
    for entry in os.scandir(root):
        if entry.is_dir(follow_symlinks=False):
            if recursive and (max_depth is None or max_depth > 0):
                yield from _scan_files(
                    entry.path, os.path.join(sub_dir, entry.name), True,
                    None if not max_depth else max_depth - 1)
        else:
            yield sub_dir, entry


def scan_files(root: str,
               sub_dir: str = '',
               pattern: str = None,
               use_regex: bool = False,
               full_path: bool = False,
               recursive: bool = False,
               max_depth: int = None) -> Iterator[str]:
    r"""Scan directory and sub-directories for files matching a pattern.

    Args:

        root: root directory
        sub_dir: restrict scan to sub directory
        pattern: return files that match pattern
        use_regex: pattern is a regular expression
        full_path: return full path
        recursive: search sub-directories
        max_depth: maximal search depth

    """
    root = os.path.expanduser(root)
    if pattern is not None and use_regex:
        pattern = re.compile(pattern)
    for subdir, file in _scan_files(
            os.path.join(root, sub_dir), sub_dir=sub_dir,
            recursive=recursive, max_depth=max_depth):
        if pattern is None \
                or use_regex and pattern.match(file.name) \
                or not use_regex and fnmatch.fnmatch(file.name, pattern):
            path = os.path.join(subdir, file.name)
            if full_path:
                path = os.path.abspath(file.path)
            yield path


def safe_path(path: Union[str, Str], *, root: str = None) -> str:
    r"""Turns ``path`` into an absolute path.

    Args:
        path: file path
        root: optional root directory

    """
    path = observe(path)
    if root:
        path = os.path.join(root, path)
    path = os.path.abspath(os.path.expanduser(path))
    return path


def assert_non_negative_number(value: Union[int, float]):
    if value < 0:
        raise ValueError('A variable that is supposed to be non-negative was '
                         'found negative.')
