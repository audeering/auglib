from typing import Sequence, Union
from functools import wraps
import ctypes
from enum import Enum, IntEnum

import numpy as np

from .api import lib
from .buffer import AudioBuffer, Transform, Source
from .utils import to_samples
from .observe import observe, Number, Bool, Float, Str


def _check_data_decorator(func):
    # Preserve docstring, see:
    # https://docs.python.org/3.6/library/functools.html#functools.wraps
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[1]
        old_ptr = ctypes.addressof(lib.AudioBuffer_data(self.obj).contents)
        old_length = lib.AudioBuffer_size(self.obj)
        func(*args, **kwargs)
        new_ptr = ctypes.addressof(lib.AudioBuffer_data(self.obj).contents)
        new_length = lib.AudioBuffer_size(self.obj)
        if old_ptr != new_ptr or old_length != new_length:
            length = lib.AudioBuffer_size(self.obj)
            self.data = np.ctypeslib.as_array(lib.AudioBuffer_data(self.obj),
                                              shape=(length,))
        return self
    return inner


class Compose(Transform):
    r"""Compose several transforms together.

    Args:
        transforms: list of transforms to compose
        bypass_prob: probability to bypass the transformation

    Example:
        >>> t = Compose([GainStage(12.0), Clip()])
        >>> with AudioBuffer(5, 16000, value=0.5, unit='samples') as buf:
        >>>     t(buf)
        [1. 1. 1. 1. 1.]

    """
    def __init__(self, transforms: Sequence[Transform], *,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.transforms = transforms

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        for t in self.transforms:
            t(buf)
        return buf


class Select(Transform):
    r"""Randomly select from a pool of transforms.

    Args:
        transforms: list of transforms to choose from
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, transforms: Sequence[Transform],
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.transforms = transforms

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        idx = np.random.randint(len(self.transforms))
        self.transforms[idx](buf)
        return buf


class Mix(Transform):
    r"""Mix the audio buffer (base) with another buffer (auxiliary) by
    adding the auxiliary buffer to the base buffer.

    Base and auxiliary buffer may differ in length but must have the
    same sampling rate.

    Mix the content of an auxiliary buffer ``aux`` on top of the base
    buffer (possibly changing its content). Individual gains can be set
    for the two signals (``gain_base_db`` and ``gain_aux_db`` expressed
    in decibels). The starting point of the mixing (with respect to the
    base buffer) can be set via the ``write_pos_base`` argument.
    Selecting a sub-segment of the auxiliary buffer is possible by means of
    ``read_pos_aux`` (the initial position of the reading pointer) and
    ``read_dur_aux`` (the total length of the selected segment). Note
    ``read_dur_aux = 0`` (default value) has the effect of selecting
    the whole auxiliary buffer. In order to force clipping of the mixed
    signal between 1.0 and -1.0, the ``clip_mix``  argument can be
    set. In order to allow the looping of the auxiliary buffer (or the
    selected sub-segment), the ``loop_aux`` argument can be used. In case
    the auxiliary buffer (or the selected sub-segment) ends beyond the
    original ending point, the extra portion will be discarded, unless
    the ``extend_base`` is set turned on, in which case the base buffer is
    extended accordingly.

    Args:
        aux: auxiliary buffer
        gain_base_db: gain of base buffer
        gain_aux_db: gain of auxiliary buffer
        write_pos_base: write position of base buffer (see ``unit``)
        read_pos_aux: read position of auxiliary buffer (see ``unit``)
        read_dur_aux: duration to read from auxiliary buffer (see
            ``unit``). Set to 0 to read the whole buffer.
        clip_mix: clip amplitude values of base buffer to the [-1, 1]
            interval (after mixing)
        loop_aux: loop auxiliary buffer if shorter than base buffer
        extend_base: if needed, extend base buffer to total required
            length (considering length of auxiliary buffer)
        unit: literal specifying the format of ``write_pos_base``,
            ``read_pos_aux`` and ``read_dur_aux``
            (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary buffer
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer(1.0, 8000) as base:
        >>>     with AudioBuffer(1.0, 8000, value=1.0) as aux:
        >>>         Mix(aux)(base)
        [1. 1. 1. ... 1. 1. 1.]

    """
    def __init__(self, aux: Union[str, Str, Source, AudioBuffer],
                 *,
                 gain_base_db: Union[float, Float] = 0.0,
                 gain_aux_db: Union[float, Float] = 0.0,
                 write_pos_base: Union[int, float, Number] = 0.0,
                 read_pos_aux: Union[int, float, Number] = 0.0,
                 read_dur_aux: Union[int, float, Number] = None,
                 clip_mix: Union[bool, Bool] = False,
                 loop_aux: Union[bool, Bool] = False,
                 extend_base: Union[bool, Bool] = False,
                 unit='seconds',
                 transform: Transform = None,
                 bypass_prob: Union[float, Float] = None):

        super().__init__(bypass_prob)
        self.aux = aux
        self.gain_base_db = gain_base_db
        self.gain_aux_db = gain_aux_db
        self.write_pos_base = write_pos_base
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux or 0
        self.clip_mix = clip_mix
        self.loop_aux = loop_aux
        self.extend_base = extend_base
        self.unit = unit
        self.transform = transform

    @_check_data_decorator
    def _mix(self, base: AudioBuffer, aux: AudioBuffer):
        write_pos_base = to_samples(self.write_pos_base, base.sampling_rate,
                                    unit=self.unit, length=len(base))
        read_pos_aux = to_samples(self.read_pos_aux, aux.sampling_rate,
                                  unit=self.unit, length=len(aux))
        read_dur_aux = to_samples(self.read_dur_aux, aux.sampling_rate,
                                  unit=self.unit, length=len(aux))
        gain_aux_db = observe(self.gain_aux_db)
        gain_base_db = observe(self.gain_base_db)
        clip_mix = observe(self.clip_mix)
        loop_aux = observe(self.loop_aux)
        extend_base = observe(self.extend_base)
        if self.transform:
            self.transform(aux)
        lib.AudioBuffer_mix(base.obj, aux.obj, gain_base_db,
                            gain_aux_db, write_pos_base, read_pos_aux,
                            read_dur_aux, clip_mix, loop_aux, extend_base)

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        if isinstance(self.aux, AudioBuffer):
            self._mix(buf, self.aux)
        elif isinstance(self.aux, Source):
            with self.aux() as aux:
                self._mix(buf, aux)
        else:
            path = observe(self.aux)
            with AudioBuffer.read(path) as aux:
                self._mix(buf, aux)
        return buf


class Append(Transform):
    r"""Append an auxiliary buffer at the end of the base buffer.

    Base and auxiliary buffer may differ in length but must have the
    same sampling rate.

    Options are provided for selecting a specific portion of the
    auxiliary buffer (see ``readPos_aux`` and ``read_dur_aux``).
    After the operation is complete, the final length of the base buffer
    will be ``read_dur_aux`` samples greater then the original length.

    Args:
        aux: auxiliary buffer
        read_pos_aux: read position of auxiliary buffer (see ``unit``)
        read_dur_aux: duration to read from auxiliary buffer (see
            ``unit``). Set to 0 to read the whole buffer.
        unit: literal specifying the format of ``read_pos_aux`` and
            ``read_dur_aux`` (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary buffer
        bypass_prob: probability to bypass the transformation

    >>> with AudioBuffer(1.0, 8000) as base:
    >>>     with AudioBuffer(1.0, 8000, value=1.0) as aux:
    >>>         Append(aux)(base)
    [0. 0. 0. ... 1. 1. 1.]

    """
    def __init__(self, aux: Union[str, Str, Source, AudioBuffer], *,
                 read_pos_aux: Union[int, float, Number] = 0.0,
                 read_dur_aux: Union[int, float, Number] = 0.0,
                 unit: str = 'seconds',
                 transform: Transform = None,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.aux = aux
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux
        self.unit = unit
        self.transform = transform

    @_check_data_decorator
    def _append(self, base: AudioBuffer, aux: AudioBuffer):
        read_pos_aux = to_samples(self.read_pos_aux,
                                  aux.sampling_rate,
                                  unit=self.unit)
        read_dur_aux = to_samples(self.read_dur_aux,
                                  aux.sampling_rate,
                                  unit=self.unit)
        if self.transform:
            self.transform(aux)
        lib.AudioBuffer_append(base.obj, aux.obj, read_pos_aux,
                               read_dur_aux)

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        if isinstance(self.aux, AudioBuffer):
            self._append(buf, self.aux)
        elif isinstance(self.aux, Source):
            with self.aux() as aux:
                self._append(buf, aux)
        else:
            path = observe(self.aux)
            with AudioBuffer.read(path) as aux:
                self._append(buf, aux)
        return buf


class AppendValue(Transform):
    r"""Expand base buffer with a constant value.

    Args:
        duration: duration to read from auxiliary buffer (see ``unit``)
        value: value to append
        unit: literal specifying the format of ``read_pos_aux`` and
            ``read_dur_aux`` (see :meth:`auglib.utils.to_samples`)
        bypass_prob: probability to bypass the transformation

    >>> with AudioBuffer(1.0, 8000) as base:
    >>>     AppendValue(1.0, value=1.0)(base)
    [0. 0. 0. ... 1. 1. 1.]

    """
    def __init__(self, duration: Union[int, float, Number],
                 value: Union[float, Float] = 0, *,
                 unit: str = 'seconds',
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.duration = duration
        self.value = value
        self.unit = unit

    @_check_data_decorator
    def call(self, buf: AudioBuffer) -> AudioBuffer:
        with AudioBuffer(duration=self.duration,
                         sampling_rate=buf.sampling_rate,
                         value=self.value, unit=self.unit) as aux:
            Append(aux)(buf)
        return buf


class Clip(Transform):
    r"""Hard/soft-clip the audio buffer.

    ``threshold`` sets the amplitude level in decibels to which the
    signal is clipped. The optional argument ``soft`` triggers a
    soft-clipping behaviour, for which the whole waveform is warped
    through a cubic non-linearity, resulting in a smooth transition
    between the flat (clipped) regions and the rest of the waveform.

    Args:
        threshold: amplitude level above which samples will be clipped (in
            decibels)
        soft: apply soft-clipping
        normalize: after clipping normalize buffer to 0 decibels
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, *,
                 threshold: Union[float, Float] = 0.0,
                 soft: Union[bool, Bool] = False,
                 normalize: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.threshold = threshold
        self.soft = soft
        self.normalize = normalize

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        threshold = observe(self.threshold)
        soft = observe(self.soft)
        normalize = observe(self.normalize)
        lib.AudioBuffer_clip(buf.obj, threshold, soft, normalize)
        return buf


class ClipByRatio(Transform):
    r"""Hard/soft-clip a certain fraction of the audio buffer.

    Rather than receiving a specific amplitude threshold, this function is
    designed to get instructed about the number of samples that are meant
    to be clipped, in relation to the total length of the signal. This
    ratio is internally translated into the amplitude threshold needed for
    achieving the specified intensity of the degradation. The optional
    argument ``soft`` triggers a soft-clipping behaviour, for which the
    whole waveform is warped through a cubic non-linearity, resulting in
    a smooth transition between the flat (clipped) regions and the
    rest of the waveform.

    Args:
        ratio: ratio between the number of samples that are to be clipped
            and the total number of samples in the buffer
        soft: apply soft-clipping
        normalize: after clipping normalize buffer to 0 decibels
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, ratio: Union[float, Float], *,
                 soft: Union[bool, Bool] = False,
                 normalize: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.ratio = ratio
        self.soft = soft
        self.normalize = normalize

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        ratio = observe(self.ratio)
        soft = observe(self.soft)
        normalize = observe(self.normalize)
        lib.AudioBuffer_clipByRatio(buf.obj, ratio, soft, normalize)
        return buf


class NormalizeByPeak(Transform):
    r"""Peak-normalize the audio buffer to a desired level.

    Args:
        peak_db: desired peak value in decibels
        clip: clip sample values to the interval [-1.0, 1.0]
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, *,
                 peak_db: Union[float, Float] = 0.0,
                 clip: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.peak_db = peak_db
        self.clip = clip

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        peak_db = observe(self.peak_db)
        clip = observe(self.clip)
        lib.AudioBuffer_normalizeByPeak(buf.obj, peak_db, clip)
        return buf


class GainStage(Transform):
    r"""Scale the buffer by the linear factor that corresponds
    to ``gain_dB`` (in decibels).

    Args:
        gain_db: amplification in decibels
        clip: clip sample values to the interval [-1.0, 1.0]
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, gain_db: Union[float, Float], *,
                 clip: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.gain_db = gain_db
        self.clip = clip

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        gain_db = observe(self.gain_db)
        clip = observe(self.clip)
        lib.AudioBuffer_gainStage(buf.obj, gain_db, clip)
        return buf


class FFTConvolve(Transform):
    r"""Convolve the audio buffer (base) with another buffer (auxiliary)
    using an impulse response (FFT-based approach).

    Args:
        aux: auxiliary buffer
        keep_tail: keep the tail of the convolution result (extending
            the length of the buffer), or to cut it out (keeping the
            original length of the input)
        transform: transformation applied to the auxiliary buffer
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, aux: Union[str, Str, Source, AudioBuffer], *,
                 keep_tail: Union[bool, Bool] = True,
                 transform: Transform = None,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.aux = aux
        self.keep_tail = keep_tail
        self.transform = transform

    def _fft_convolve(self, base: AudioBuffer, aux: AudioBuffer):
        if self.transform:
            self.transform(aux)
        keep_tail = observe(self.keep_tail)
        lib.AudioBuffer_fftConvolve(base.obj, aux.obj, keep_tail)

    @_check_data_decorator
    def call(self, buf: AudioBuffer) -> AudioBuffer:
        if isinstance(self.aux, AudioBuffer):
            self._fft_convolve(buf, self.aux)
        elif isinstance(self.aux, Source):
            with self.aux() as aux:
                self._fft_convolve(buf, aux)
        else:
            path = observe(self.aux)
            with AudioBuffer.read(path) as aux:
                self._fft_convolve(buf, aux)
        return buf


class FilterDesign(Enum):
    r"""
    * `BUTTERWORTH`: Butterworth filter design
    """
    BUTTERWORTH = 'butter'


class LowPass(Transform):
    r"""Run audio buffer through a low-pass filter.

    Args:
        cutoff: cutoff frequency in Hz
        order: filter order
        design: filter design (see :class:`FilterDesign`)
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, cutoff: Union[float, Float], *,
                 order: int = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.cutoff = cutoff
        self.order = order
        self.design = design

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        cutoff = observe(self.cutoff)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthLowPassFilter(buf.obj, cutoff,
                                                     self.order)
        else:
            assert False, 'unknown filter design {}'.format(self.design)
        return buf


class HighPass(Transform):
    r"""Run audio buffer through a high-pass filter.

    Args:
        cutoff: cutoff frequency in Hz
        order: filter order
        design: filter design (see :class:`FilterDesign`)
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, cutoff: Union[float, Float], *,
                 order: int = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.cutoff = cutoff
        self.order = order
        self.design = design

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        cutoff = observe(self.cutoff)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthHighPassFilter(buf.obj, cutoff,
                                                      self.order)
        else:
            assert False, 'unknown filter design {}'.format(self.design)
        return buf


class BandPass(Transform):
    r"""Run audio buffer through a band-pass filter.

    Args:
        center: center frequency in Hz
        bandwidth: bandwidth frequency in Hz
        order: filter order
        design: filter design (see :class:`FilterDesign`)
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, center: Union[float, Float],
                 bandwidth: Union[float, Float], *,
                 order: int = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.center = center
        self.bandwidth = bandwidth
        self.order = order
        self.design = design

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        center = observe(self.center)
        bandwidth = observe(self.bandwidth)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthBandPassFilter(buf.obj, center,
                                                      bandwidth, self.order)
        else:
            assert False, 'unknown filter design {}'.format(self.design)
        return buf


class WhiteNoiseUniform(Transform):
    r"""Adds uniform white noise.

    Args:
        gain_db: gain in decibels
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, *,
                 gain_db: Union[float, Float] = 0.0,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.gain_db = gain_db

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        gain_db = observe(self.gain_db)
        lib.AudioBuffer_addWhiteNoiseUniform(buf.obj, gain_db)
        return buf


class WhiteNoiseGaussian(Transform):
    r"""Adds Gaussian white noise.

    Args:
        gain_db: gain in decibels
        stddev: standard deviation
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, *,
                 gain_db: Union[float, Float] = 0.0,
                 stddev: Union[float, Float] = 0.3,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.gain_db = gain_db
        self.stddev = stddev

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        gain_db = observe(self.gain_db)
        stddev = observe(self.stddev)
        lib.AudioBuffer_addWhiteNoiseGaussian(buf.obj, gain_db, stddev)
        return buf


class PinkNoise(Transform):
    r"""Adds pink noise.

    Args:
        gain_db: gain in decibels
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, *,
                 gain_db: Union[float, Float] = 0.0,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.gain_db = gain_db

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        gain_db = observe(self.gain_db)
        lib.AudioBuffer_addPinkNoise(buf.obj, gain_db)
        return buf


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


class Tone(Transform):
    r"""Adds basic waveform.

    Args:
        freq: fundamental frequency in Hz
        gain_db: gain in decibels
        shape: tone shape (see :class:`ToneShape`)
        lfo_rate: modulation rate of Low Frequency Oscillator
        lfo_range: modulation range of Low Frequency Oscillator
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, freq: Union[float, Float],
                 *, gain_db: Union[float, Float] = 0.0,
                 shape: ToneShape = ToneShape.SINE,
                 lfo_rate: Union[float, Float] = 0.0,
                 lfo_range: Union[float, Float] = 0.0,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.freq = freq
        self.gain_db = gain_db
        self.shape = shape
        self.lfo_rate = lfo_rate
        self.lfo_range = lfo_range

    def call(self, buf: AudioBuffer) -> AudioBuffer:
        freq = observe(self.freq)
        gain_db = observe(self.gain_db)
        lfo_rate = observe(self.lfo_rate)
        lfo_range = observe(self.lfo_range)
        lib.AudioBuffer_addTone(buf.obj, freq, gain_db, self.shape.value,
                                lfo_rate, lfo_range)
        return buf
