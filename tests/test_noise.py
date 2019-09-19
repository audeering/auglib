import pytest
import numpy as np

from auglib import random_seed, WhiteNoiseUniform, WhiteNoiseGaussian, \
    PinkNoise, AudioBuffer
from auglib.api import lib


@pytest.mark.parametrize('seed',
                         [1234])
def test_random_generator(seed):

    sr = 8000
    dur = 1.0

    random_seed(seed)
    noise = WhiteNoiseUniform(dur, sr)
    random_seed(seed)
    noise2 = WhiteNoiseUniform(dur, sr)
    assert noise == noise2
    noise2.free()
    random_seed(seed + 1)
    noise2 = WhiteNoiseUniform(dur, sr)
    assert noise != noise2
    noise2.free()
    noise.free()


@pytest.mark.parametrize('dur,sr,gain,seed',
                         [(1.0, 8000, 0.0, 1)])
def test_WhiteNoiseUniform(dur, sr, gain, seed):

    random_seed(seed)
    noise = WhiteNoiseUniform(dur, sr, gain_db=gain)
    buf = AudioBuffer(len(noise), noise.sampling_rate, unit='samples')
    random_seed(seed)
    lib.AudioBuffer_addWhiteNoiseUniform(buf.obj, gain)
    np.testing.assert_equal(noise.data, buf.data)
    buf.free()
    noise.free()


@pytest.mark.parametrize('dur,sr,gain,stddev,seed',
                         [(1.0, 8000, 0.0, 0.3, 1)])
def test_WhiteNoiseGaussian(dur, sr, gain, stddev, seed):

    random_seed(seed)
    noise = WhiteNoiseGaussian(dur, sr, stddev=stddev, gain_db=gain)
    buf = AudioBuffer(len(noise), noise.sampling_rate, unit='samples')
    random_seed(seed)
    lib.AudioBuffer_addWhiteNoiseGaussian(buf.obj, gain, stddev)
    np.testing.assert_equal(noise.data, buf.data)
    buf.free()
    noise.free()


@pytest.mark.parametrize('dur,sr,gain,seed',
                         [(1.0, 8000, 0.0, 1)])
def test_PinkNoise(dur, sr, gain, seed):

    random_seed(seed)
    noise = PinkNoise(dur, sr, gain_db=gain)
    buf = AudioBuffer(len(noise), noise.sampling_rate, unit='samples')
    random_seed(seed)
    lib.AudioBuffer_addPinkNoise(buf.obj, gain)
    np.testing.assert_equal(noise.data, buf.data)
    buf.free()
    noise.free()
