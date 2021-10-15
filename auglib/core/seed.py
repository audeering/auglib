import random

import numpy as np

from auglib.core.api import lib


def seed(seed: int):
    r"""(Re-)initialize random generator.

    The random generator is shared among all audio classes.

    Args:
        seed: seed number

    """

    random.seed(seed)
    np.random.seed(seed)
    if seed == 0:
        # 0 is random initialization in the C library
        # so we switch to another value.
        # We use max integer - 1,
        # see https://stackoverflow.com/a/7604981
        seed = 9223372036854775806
    lib.auglib_random_seed(seed)
