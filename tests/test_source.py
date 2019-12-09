import pytest

from auglib.utils import random_seed
from auglib import AudioBuffer
from auglib.transform import WhiteNoiseUniform


@pytest.mark.parametrize('seed',
                         [1234])
def test_random_generator(seed):

    sr = 8000
    dur = 1.0

    random_seed(seed)
    with WhiteNoiseUniform()(AudioBuffer(dur, sr)) as noise:
        random_seed(seed)
        with WhiteNoiseUniform()(AudioBuffer(dur, sr)) as noise2:
            assert noise == noise2
        random_seed(seed + 1)
        with WhiteNoiseUniform()(AudioBuffer(dur, sr)) as noise2:
            assert noise != noise2
