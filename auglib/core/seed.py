import random

import numpy as np


SEED = 0


def get_seed() -> int:
    r"""Get current state of seed."""
    global SEED
    return SEED


def seed(seed: int):
    r"""(Re-)initialize random generator.

    The random generator is shared among all audio classes.

    Args:
        seed: seed number

    """
    global SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
