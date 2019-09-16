from enum import IntEnum

from .api import lib
from .buffer import AudioBuffer


class ToneShape(IntEnum):
    r"""
    * `SINE`: sine wave
    * `SQUARE`: square wave
    * `TRIANGLE`: triangle wave
    * `SAWTOOTH`: sawtooth wave
    """
    SINE = 0
    SQUARE = 1
    TRIANGLE = 2
    SAWTOOTH = 3


class Tone(AudioBuffer):
    r"""Creates a :class:`AudioBuffer` initialized with a basic waveform.

    Args:
        duration: buffer duration (see ``unit``)
        sampling_rate: sampling rate in Hz
        shape: tone shape (see :class:`ToneShape`)
        freq: fundamental frequency in Hz
        lfo_rate: modulation rate of Low Frequency Oscillator
        lfo_range: modulation range of Low Frequency Oscillator
        gain_db: gain in decibel
        unit: literal specifying the format of ``duration``
             (see :meth:`auglib.utils.dur2samples`)

    """
    def __init__(self, duration: float, sampling_rate: int, *,
                 shape: ToneShape = ToneShape.SINE, freq: float = 440.0,
                 lfo_rate: float = 0.0, lfo_range: float = 0.0,
                 gain_db: float = 0.0, unit='seconds'):
        super().__init__(duration, sampling_rate, unit=unit)
        lib.AudioBuffer_addTone(self.obj, shape.value, freq, lfo_rate,
                                lfo_range, gain_db)
