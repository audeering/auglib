import pytest

import auglib
from auglib import AudioBuffer
from auglib.transform import WhiteNoiseUniform


@pytest.mark.parametrize('seed',
                         [1234])
def test_random_generator(seed):

    sr = 8000
    dur = 1.0

    auglib.seed(seed)
    with WhiteNoiseUniform()(AudioBuffer(dur, sr)) as noise:
        auglib.seed(seed)
        with WhiteNoiseUniform()(AudioBuffer(dur, sr)) as noise2:
            assert noise == noise2
        auglib.seed(seed + 1)
        with WhiteNoiseUniform()(AudioBuffer(dur, sr)) as noise2:
            assert noise != noise2
