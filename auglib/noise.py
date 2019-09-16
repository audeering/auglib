from .api import lib
from .buffer import AudioBuffer


def random_seed(seed: int = 0):
    r"""(Re-)initialize random generator..

    Note: controls a random generator that is shared among all noise classes.

    Args:
        seed: seed number (0 for random initialization)

    """
    lib.auglib_random_seed(seed)


class WhiteNoiseUniform(AudioBuffer):
    r"""Creates a :class:`AudioBuffer` initialized with uniform
    white noise.

    Args:
        duration: buffer duration (see ``unit``)
        sampling_rate: sampling rate in Hz
        gain_db: gain in decibel
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.dur2samples`)

    """
    def __init__(self, duration: float, sampling_rate: int, *,
                 gain_db: float = 0.0, unit='seconds'):
        super().__init__(duration, sampling_rate, unit=unit)
        lib.AudioBuffer_addWhiteNoiseUniform(self.obj, gain_db)


class WhiteNoiseGaussian(AudioBuffer):
    r"""Creates a :class:`AudioBuffer` initialized with Gaussian white noise.

    Args:
        duration: buffer duration (see ``unit``)
        sampling_rate: sampling rate in Hz
        gain_db: gain in decibel
        stddev: standard deviation
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.dur2samples`)

    """
    def __init__(self, duration: float, sampling_rate: int, *,
                 gain_db: float = 0.0, stddev: float = 0.3, unit='seconds'):
        super().__init__(duration, sampling_rate, unit=unit)
        lib.AudioBuffer_addWhiteNoiseGaussian(self.obj, gain_db, stddev)


class PinkNoise(AudioBuffer):
    r"""Creates a :class:`AudioBuffer` initialized with pink noise.

    Args:
        duration: buffer duration (see ``unit``)
        sampling_rate: sampling rate in Hz
        gain_db: gain in decibel
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.dur2samples`)

    """
    def __init__(self, duration: float, sampling_rate: int, *,
                 gain_db: float = 0.0, unit='seconds'):
        super().__init__(duration, sampling_rate, unit=unit)
        lib.AudioBuffer_addPinkNoise(self.obj, gain_db)
