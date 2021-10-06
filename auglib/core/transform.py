from typing import Callable, Optional, Sequence, Union
from functools import wraps
import ctypes
from enum import Enum, IntEnum

import numpy as np

import audobject

from auglib.core.api import lib
from auglib.core.buffer import (
    AudioBuffer,
    Source,
    Transform,
)
from auglib.core.utils import to_samples
from auglib.core.observe import (
    Bool,
    Float,
    Int,
    Number,
    observe,
    Str,
)


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
        ...     t(buf)
        array([1., 1., 1., 1., 1.], dtype=float32)

    """
    def __init__(self, transforms: Sequence[Transform], *,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.transforms = transforms

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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
    ``read_dur_aux = None`` (default value) has the effect of selecting
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
            ``unit``). Set to None to read the whole buffer.
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
        ...     with AudioBuffer(1.0, 8000, value=1.0) as aux:
        ...         Mix(aux)(base)
        array([1., 1., 1., ..., 1., 1., 1.], dtype=float32)

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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    Example:

        >>> with AudioBuffer(1.0, 8000) as base:
        ...     with AudioBuffer(1.0, 8000, value=1.0) as aux:
        ...         Append(aux)(base)
        array([0., 0., 0., ..., 1., 1., 1.], dtype=float32)

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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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
        unit: Literal specifying the format of ``duration`` (see
            :meth:`auglib.utils.to_samples`).
        bypass_prob: probability to bypass the transformation

    Example:

        >>> with AudioBuffer(1.0, 8000) as base:
        ...     AppendValue(1.0, value=1.0)(base)
        array([0., 0., 0., ..., 1., 1., 1.], dtype=float32)

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
    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        with AudioBuffer(duration=self.duration,
                         sampling_rate=buf.sampling_rate,
                         value=self.value, unit=self.unit) as aux:
            Append(aux)(buf)
        return buf


class Trim(Transform):
    r"""Trims base buffer to desired duration, given a desired starting point.

    Args:
        start_pos: Starting point of the trimmed region, relative to the input
            buffer (see ``unit``).
        duration: Target duration of the resulting buffer (see ``unit``). If
            set to None, the selected section extends until the end of the
            original buffer.
        unit: Literal specifying the format of ``start`` and ``duration`` (see
            :meth:`auglib.utils.to_samples`).
        bypass_prob: Probability to bypass the transformation.

    Example:

        >>> with AudioBuffer(0.5, 8000) as buf:
        ...     AppendValue(0.5, value=1.0)(buf)
        ...     Trim(start_pos=0.5)(buf)
        array([0., 0., 0., ..., 1., 1., 1.], dtype=float32)
        array([1., 1., 1., ..., 1., 1., 1.], dtype=float32)

    """
    def __init__(self,
                 *,
                 start_pos: Union[int, float, Number] = 0,
                 duration: Union[int, float, Number] = None,
                 unit: str = 'seconds',
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.start_pos = start_pos
        self.duration = duration or 0
        self.unit = unit

    @_check_data_decorator
    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        start_pos = to_samples(observe(self.start_pos), buf.sampling_rate,
                               unit=self.unit, length=len(buf))
        duration = to_samples(observe(self.duration), buf.sampling_rate,
                              unit=self.unit, length=len(buf))
        lib.AudioBuffer_trim(buf.obj, start_pos, duration)
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        peak_db = observe(self.peak_db)
        clip = observe(self.clip)
        lib.AudioBuffer_normalizeByPeak(buf.obj, peak_db, clip)
        return buf


class GainStage(Transform):
    r"""Scale the buffer by the linear factor that corresponds
    to ``gain_dB`` (in decibels).

    .. note:: If ``max_peak_db`` is not ``None`` and the resulting peak level
        exceeds the given value, the actual gain is adjusted so that the
        peak level of the output matches the specified ceiling level. This
        means that the final gain applied might be different to that
        specified. Also note that if ``max_peak_db`` > 0dB, ``clip`` is
        ``True``, and the peak resulting from scaling the signal is greater
        than 0dB, then the final peak level is actually forced to 0dB (with
        clipping)

    Args:
        gain_db: amplification in decibels
        max_peak_db: maximum peak level allowed in decibels (see note)
        clip: clip sample values to the interval [-1.0, 1.0]
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, gain_db: Union[float, Float], *,
                 max_peak_db: Union[float, Float] = None,
                 clip: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.gain_db = gain_db
        self.max_peak_db = max_peak_db
        self.clip = clip

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        gain_db = observe(self.gain_db)
        clip = observe(self.clip)
        max_peak_db = observe(self.max_peak_db)
        if max_peak_db is not None:
            lib.AudioBuffer_gainStageSafe(buf.obj, gain_db, max_peak_db)
            if self.clip:
                lib.AudioBuffer_clip(buf.obj, 0.0, False, False)
        else:
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
    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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
                 order: Union[int, Int] = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.cutoff = cutoff
        self.order = order
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        cutoff = observe(self.cutoff)
        order = observe(self.order)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthLowPassFilter(buf.obj, cutoff, order)
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
                 order: Union[int, Int] = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.cutoff = cutoff
        self.order = order
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        cutoff = observe(self.cutoff)
        order = observe(self.order)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthHighPassFilter(buf.obj, cutoff, order)
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
                 order: Union[int, Int] = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.center = center
        self.bandwidth = bandwidth
        self.order = order
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        center = observe(self.center)
        bandwidth = observe(self.bandwidth)
        order = observe(self.order)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthBandPassFilter(buf.obj, center,
                                                      bandwidth, order)
        else:
            assert False, 'unknown filter design {}'.format(self.design)
        return buf


class BandStop(Transform):
    r"""Run audio buffer through a band-stop filter.

    Args:
        center: center frequency in Hz
        bandwidth: bandwidth frequency in Hz
        order: filter order
        design: filter design (see :class:`FilterDesign`)
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, center: Union[float, Float],
                 bandwidth: Union[float, Float], *,
                 order: Union[int, Int] = 1,
                 design: str = FilterDesign.BUTTERWORTH.value,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.center = center
        self.bandwidth = bandwidth
        self.order = order
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        center = observe(self.center)
        bandwidth = observe(self.bandwidth)
        order = observe(self.order)
        if self.design == FilterDesign.BUTTERWORTH.value:
            lib.AudioBuffer_butterworthBandStopFilter(buf.obj, center,
                                                      bandwidth, order)
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
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

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        freq = observe(self.freq)
        gain_db = observe(self.gain_db)
        lfo_rate = observe(self.lfo_rate)
        lfo_range = observe(self.lfo_range)
        lib.AudioBuffer_addTone(buf.obj, freq, gain_db, self.shape.value,
                                lfo_rate, lfo_range)
        return buf


class CompressDynamicRange(Transform):
    r"""Compress the dynamic range of the buffer by attenuating any sample
    whose amplitude exceeds a certain ``threshold_db``.
    The intensity of the attenuation is determined by the ``ratio`` parameter
    (the higher the ratio, the stronger the gain reduction). To avoid heavy
    distortion, the gain reduction is smoothed over time with a contour that is
    governed by the ``attack_time`` and the ``release_time`` parameters.

    The input-output characteristic also features a non-linear region ("knee")
    around the threshold. The width of this region is controlled by the
    ``knee_radius_db`` parameter (expressed in decibels, and in absolute
    value): the nonlinear region is entered when the input signal exceeds a
    level given by ``threshold_db - kneeRadius_db``, hence some gain
    reduction can be also seen before hitting the main threshold, if the
    knee radius is greater than zero.

    Optionally, the resulting signal can be amplified (linearly) by means of
    the ``makeup_dB`` gain parameter (expressed in decibels). Sample values can
    also be clipped to the interval ``[-1.0, 1.0]`` when exceeding this range:
    this behaviour is achieved by setting the ``clip`` argument.

    .. note:: Setting makeup_dB to None triggers a special behaviour,
        for which the makeup gain is computed automatically in a way that
        the peak level of the processed signal is equal to the original peak
        level (before compression).

    Args:
        threshold_db: threshold in decibels
        ratio: ratio (the higher the ratio, the stronger the gain reduction)
        attack_time: attack time in seconds
        release_time: release time in seconds
        makeup_db: optional amplification gain
        clip: clip signal
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self,
                 threshold_db: Union[float, Float],
                 ratio: Union[float, Float], *,
                 attack_time: Union[float, Float] = 0.01,
                 release_time: Union[float, Float] = 0.02,
                 knee_radius_db: Union[float, Float] = 4.0,
                 makeup_db: Union[None, float, Float] = 0.0,
                 clip: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_time = attack_time
        self.release_time = release_time
        self.knee_radius_db = knee_radius_db
        self.makeup_db = makeup_db
        self.clip = clip

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        threshold_db = observe(self.threshold_db)
        ratio = observe(self.ratio)
        attack_time = observe(self.attack_time)
        release_time = observe(self.release_time)
        knee_radius_db = observe(self.knee_radius_db)
        if self.makeup_db is None:
            makeup_db = np.nan
        else:
            makeup_db = observe(self.makeup_db)
        clip = observe(self.clip)
        lib.AudioBuffer_compressDynamicRange(buf.obj,
                                             threshold_db,
                                             ratio,
                                             attack_time,
                                             release_time,
                                             knee_radius_db,
                                             makeup_db,
                                             clip)
        return buf


class AMRNB(Transform):
    r"""Encode-decode the buffer using Adaptive Multi-Rate (AMR) lossy codec
    (Narrow Band version).

    .. note:: The input signal must be narrow-band (it must be sampled at
        8kHz).

    .. note:: Supported bit rates: 4750, 5150, 5900, 6700, 7400, 7950, 10200,
        12200. Any positive bit rate is allowed, but it wil be internally
        converted to the closest among those listed above.

    Args:
        bit_rate: target bit rate of the encoded stream (in bits per second)
        dtx: enable discontinuous transmission (DTX)
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self,
                 bit_rate: Union[int, Int],
                 *,
                 dtx: Union[bool, Bool] = False,
                 bypass_prob: Union[float, Float] = None):
        super().__init__(bypass_prob)
        self.bit_rate = bit_rate
        self.dtx = dtx

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        bit_rate = observe(self.bit_rate)
        dtx = observe(self.dtx)
        lib.AudioBuffer_AMRNB(buf.obj, bit_rate, 1 if dtx else 0)
        return buf


class Function(Transform):
    r"""Apply a custom function to the buffer.

    The function gets as input a :class:`numpy.ndarray`
    of shape ``(channels, samples)``
    with the content of the audio buffer
    and the sampling rate.
    I can either return ``None``
    if the operation is applied in-place,
    or a new :class:`numpy.ndarray`
    that will be written back to the buffer.
    If necessary,
    the size of the audio buffer will be shrinked or expanded
    to fit the returned array.

    Note that the object is not serializable
    if the function relies on other locally defined functions.
    For instance,
    in this example object ``f`` is not serializable:

    .. code-block:: python

        def _plus_1(x, sr):
            return x + 1

        def plus_1(x, sr):
            return _plus_1(x, sr)  # calls local function -> not serializable

        f = auglib.transform.Function(plus_1)


    Args:
        function: (lambda) function object
        bypass_prob: probability to bypass the transformation

    Example:
        >>> buf = AudioBuffer(10, 8000, unit='samples')
        >>> buf
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
        >>> def plus_1(x, sr):
        ...     x += 1
        >>> Function(plus_1)(buf)
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
        >>> def halve(x, sr):
        ...     return x[:, ::2]
        >>> Function(halve)(buf)
        array([1., 1., 1., 1., 1.], dtype=float32)
        >>> Function(lambda x, sr: x * 2)(buf)
        array([2., 2., 2., 2., 2.], dtype=float32)
        >>> buf.free()

    """
    @audobject.init_decorator(
        resolvers={
            'function': audobject.FunctionResolver,
        },
    )
    def __init__(
            self,
            function: Callable[
                [np.ndarray, int],
                Optional[np.ndarray],
            ],
            *,
            bypass_prob: Union[float, Float] = None,
    ):
        super().__init__(bypass_prob)
        self.function = function

    def _call(self, buf: AudioBuffer):

        # apply function
        x = buf.data.reshape((1, -1))
        y = self.function(x, buf.sampling_rate)

        # if a new array is returned
        # we copy the result to the buffer,
        # otherwise we assume an inplace operation
        if y is not None:
            # ensure result has correct data type
            y = y.astype(x.dtype)
            # if necessary fit buffer size to result
            if y.size < x.size:
                Trim(
                    duration=y.size,
                    unit='samples',
                )(buf)
            elif y.size > x.size:
                AppendValue(
                    duration=y.size - x.size,
                    unit='samples',
                )(buf)
            # copy result to buffer
            buf.data[:] = y
