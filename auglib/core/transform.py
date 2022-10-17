import ctypes
from functools import wraps
import typing
from typing import Callable, Optional, Sequence, Union

import numpy as np

import audeer
import audobject
import audresample

from auglib.core.api import lib
from auglib.core.buffer import (
    AudioBuffer,
    rms_db,
)
from auglib.core import observe
from auglib.core.exception import _check_exception_decorator
from auglib.core.resolver import AudioBufferResolver
from auglib.core.seed import seed
from auglib.core.time import Time
from auglib.core.utils import (
    from_db,
    to_db,
    to_samples,
)


SUPPORTED_FILL_STRATEGIES = ['none', 'zeros', 'loop']
SUPPORTED_FILL_POSITIONS = ['right', 'left', 'both']
SUPPORTED_FILTER_DESIGNS = ['butter']
SUPPORTED_TONE_SHAPES = ['sine', 'square', 'triangle', 'sawtooth']


def buffer_length_can_change_decorator(func):
    r"""Decorator to adjust the buffer length in C library.

    Add this decorator to ``_call()`` methods
    that can potentially change the length
    of the base buffer.

    """
    # Wrap func to preserve docstring, see:
    # https://docs.python.org/3.6/library/functools.html#functools.wraps
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[1]
        func(*args, **kwargs)
        # Adjust length of buffer to desired length after transform
        length = lib.AudioBuffer_size(self._obj)
        self._data = np.ctypeslib.as_array(
            lib.AudioBuffer_data(self._obj),
            shape=(length,),
        )
        return self
    return inner


def get_noise_gain_from_requested_snr(rms_signal_db, rms_noise_db, snr_db):
    r"""Translates requested SNR to gain of noise signal.

    Args:
        rms_signal_db: root mean square of signal in dB
        rms_noise_db: root mean square of noise signal
            with max amplitude in dB
        snr_db: desired signal-to-noise ration in dB

    Returns:
        gain to be applied to noise signal
            to achieve desired SNR in dB

    """
    # SNR = RMS_signal^2 / (gain * RMS_noise)^2
    # => SNR_dB = 10 log10(RMS_signal^2 / (gain * RMS_noise)^2)
    # => SNR_dB = 10 log10(RMS_signal^2) - 10 log10((gain * RMS_noise)^2)
    # => SNR_dB = 20 log10(RMS_signal) - 20 log10(gain * RMS_noise)
    # => SNR_dB = 20 log10(RMS_signal) - 20 log10(gain) - 20 log10(RMS_noise)
    # => SNR_dB = RMS_signal_dB - gain_dB - RMS_noise_dB
    gain_db = rms_signal_db - rms_noise_db - snr_db
    return gain_db


class Base(audobject.Object):
    r"""Base class for transforms applied to an :class:`auglib.AudioBuffer`.

    Args:
        bypass_prob: probability to bypass the transformation
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        aux: auxiliary buffer,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a buffer with the same length
            as the base buffer
            containing zeros
        transform: transformation applied to the auxiliary buffer

    """
    @audobject.init_decorator(
        resolvers={
            'aux': AudioBufferResolver,
        }
    )
    def __init__(
            self,
            bypass_prob: Union[float, observe.Base] = None,
            *,
            preserve_level: Union[bool, observe.Base] = False,
            aux: Union[str, observe.Base, AudioBuffer, 'Base'] = None,
            transform: 'Base' = None,
            num_repeat: int = None,
    ):
        self.bypass_prob = bypass_prob
        self.preserve_level = preserve_level
        self.aux = aux
        self.transform = transform
        self.num_repeat = num_repeat

    def _call(
            self,
            buf: AudioBuffer,
            aux: AudioBuffer = None,
    ):  # pragma: no cover
        r"""Transform an :class:`auglib.AudioBuffer`.

        Args:
            buf: audio buffer

        Raises:
            NotImplementedError: raised if not overwritten in child class

        """
        raise NotImplementedError()

    @_check_exception_decorator
    def __call__(self, buf: AudioBuffer) -> AudioBuffer:

        bypass_prob = observe.observe(self.bypass_prob)
        preserve_level = observe.observe(self.preserve_level)
        if (
                bypass_prob is None
                or np.random.random_sample() >= bypass_prob
        ):
            if preserve_level:
                base_level = rms_db(buf._data)
            num_repeat = observe.observe(self.num_repeat) or 1
            for _ in range(num_repeat):
                if self.aux is None:
                    self._call(buf)
                else:
                    free_aux = False
                    aux = observe.observe(self.aux)
                    if isinstance(aux, str):
                        aux = AudioBuffer.read(aux)
                        free_aux = True
                    elif isinstance(aux, Base):
                        # assume aux is transform
                        generator = aux
                        aux = AudioBuffer(buf.duration, buf.sampling_rate)
                        generator(aux)
                        free_aux = True
                    if self.transform is not None:
                        self.transform(aux)
                    self._call(buf, aux)
                    if free_aux:
                        aux.free()
            if preserve_level:
                mix_level = rms_db(buf._data)
                gain = from_db(base_level - mix_level)
                buf._data = gain * buf._data
        return buf


class AMRNB(Base):
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
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer(0.5, 8000, unit='seconds', value=0.5) as buf:
        ...     AMRNB(7400)(buf).peak
        0.02783203125

    """
    def __init__(
            self,
            bit_rate: Union[int, observe.Base],
            *,
            dtx: Union[bool, observe.Base] = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.bit_rate = bit_rate
        self.dtx = dtx

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        bit_rate = observe.observe(self.bit_rate)
        dtx = observe.observe(self.dtx)
        lib.AudioBuffer_AMRNB(buf._obj, bit_rate, 1 if dtx else 0)
        return buf


class Append(Base):
    r"""Append an auxiliary buffer to the base buffer.

    Base and auxiliary buffer may differ in length but must have the
    same sampling rate.

    Args:
        aux: auxiliary buffer,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a buffer with the same length
            as the base buffer
            containing zeros
        read_pos_aux: read position of auxiliary buffer (see ``unit``)
        read_dur_aux: duration to read from auxiliary buffer
            (see ``unit``).
            Set to ``None`` or ``0`` to read the whole buffer
        unit: literal specifying the format of ``read_pos_aux`` and
            ``read_dur_aux`` (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary buffer
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     with AudioBuffer.from_array([5, 6], 8000) as aux:
        ...         Append(aux)(buf)
        array([[1., 2., 3., 4., 5., 6.]], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, AudioBuffer, Base],
            *,
            read_pos_aux: Union[int, float, observe.Base, Time] = 0.0,
            read_dur_aux: Union[int, float, observe.Base, Time] = None,
            unit: str = 'seconds',
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
            aux=aux,
            transform=transform,
        )
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux or 0
        self.unit = unit

    @buffer_length_can_change_decorator
    def _call(self, base: AudioBuffer, aux: AudioBuffer) -> AudioBuffer:
        read_pos_aux = to_samples(self.read_pos_aux,
                                  sampling_rate=aux.sampling_rate,
                                  unit=self.unit)
        read_dur_aux = to_samples(self.read_dur_aux,
                                  sampling_rate=aux.sampling_rate,
                                  unit=self.unit)
        lib.AudioBuffer_append(base._obj, aux._obj, read_pos_aux,
                               read_dur_aux)
        return base


class AppendValue(Base):
    r"""Expand base buffer with a constant value.

    Args:
        duration: duration of buffer with constant value
            that will be appended
            (see ``unit``)
        value: value to append
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`).
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     AppendValue(2, 5, unit='samples')(buf)
        array([[1., 2., 3., 4., 5., 5.]], dtype=float32)

    """
    def __init__(
            self,
            duration: Union[int, float, observe.Base, Time],
            value: Union[float, observe.Base] = 0,
            *,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.duration = duration
        self.value = value
        self.unit = unit

    @buffer_length_can_change_decorator
    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        if self.duration != 0:
            with AudioBuffer(duration=self.duration,
                             sampling_rate=buf.sampling_rate,
                             value=self.value, unit=self.unit) as aux:
                Append(aux)(buf)
        return buf


class BabbleNoise(Base):
    """Adds Babble Noise.

    Babble noise refers to having several speakers
    in the background
    all talking at the same time.

    :class:`BabbleNoise` does not use built-in speech signals
    but expects a sequence of speech buffers or files
    as ``speech`` argument,
    from which it then randomly samples the speech.

    Args:
        speech: speech buffer(s) used to create babble noise
        num_speakers: number of speech buffers
            used to create babble noise.
            If not enough speech buffers are given
            it will repeat all
            or some of them
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> seed(0)
        >>> with AudioBuffer.from_array([0, 0, 0, 0], 8000) as base:
        ...     with AudioBuffer.from_array([1, 0, 0], 8000) as aux:
        ...         BabbleNoise([aux], num_speakers=2)(base)
        array([[0. , 0.5, 0.5, 0. ]], dtype=float32)

    """
    def __init__(
            self,
            speech: Sequence[Union[str, AudioBuffer]],
            *,
            num_speakers: Union[int, observe.Base] = 5,
            gain_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.speech = speech
        self.num_speakers = num_speakers
        self.gain_db = gain_db
        self.snr_db = snr_db

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        with AudioBuffer(
                len(buf),
                buf.sampling_rate,
                unit='samples',
        ) as babble:
            # First create buffer containing babble noise
            # by summing speech buffers
            num_repeat = observe.observe(self.num_speakers)
            transform = Mix(
                observe.List(self.speech, draw=True),
                gain_aux_db=to_db(1 / num_repeat),
                num_repeat=num_repeat,
                loop_aux=True,
                # Cycle the input signal
                transform=Shift(observe.FloatUni(0, babble.duration)),
            )
            transform(babble)
            # Mix the babble noise to aux buffer
            transform = Mix(
                babble,
                gain_aux_db=self.gain_db,
                snr_db=self.snr_db,
            )
            transform(buf)
        return buf


class BandPass(Base):
    r"""Run audio buffer through a band-pass filter.

    Args:
        center: center frequency in Hz
        bandwidth: bandwidth frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Example:
        >>> with AudioBuffer.from_array([1, 2], 8000) as buf:
        ...     BandPass(2000, 1000)(buf)
        array([[0.2928932 , 0.58578646]], dtype=float32)

    """
    def __init__(
            self,
            center: Union[float, observe.Base],
            bandwidth: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.center = center
        self.bandwidth = bandwidth
        self.order = order
        if design not in SUPPORTED_FILTER_DESIGNS:
            raise ValueError(
                f"Unknown filter design '{design}'. "
                "Supported designs are: "
                f"{', '.join(SUPPORTED_FILTER_DESIGNS)}."
            )
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        center = observe.observe(self.center)
        bandwidth = observe.observe(self.bandwidth)
        order = observe.observe(self.order)
        if self.design == 'butter':
            lib.AudioBuffer_butterworthBandPassFilter(buf._obj, center,
                                                      bandwidth, order)
        return buf


class BandStop(Base):
    r"""Run audio buffer through a band-stop filter.

    Args:
        center: center frequency in Hz
        bandwidth: bandwidth frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Example:
        >>> with AudioBuffer.from_array([1, 2], 8000) as buf:
        ...     BandStop(2000, 1000)(buf)
        array([[0.70710677, 1.4142135 ]], dtype=float32)

    """
    def __init__(
            self,
            center: Union[float, observe.Base],
            bandwidth: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.center = center
        self.bandwidth = bandwidth
        self.order = order
        if design not in SUPPORTED_FILTER_DESIGNS:
            raise ValueError(
                f"Unknown filter design '{design}'. "
                "Supported designs are: "
                f"{', '.join(SUPPORTED_FILTER_DESIGNS)}."
            )
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        center = observe.observe(self.center)
        bandwidth = observe.observe(self.bandwidth)
        order = observe.observe(self.order)
        if self.design == 'butter':
            lib.AudioBuffer_butterworthBandStopFilter(buf._obj, center,
                                                      bandwidth, order)
        return buf


class Clip(Base):
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
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Clip(threshold=to_db(2))(buf)
        array([[1., 2., 2., 2.]], dtype=float32)

    """
    def __init__(
            self,
            *,
            threshold: Union[float, observe.Base] = 0.0,
            soft: Union[bool, observe.Base] = False,
            normalize: Union[bool, observe.Base] = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.threshold = threshold
        self.soft = soft
        self.normalize = normalize

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        threshold = observe.observe(self.threshold)
        soft = observe.observe(self.soft)
        normalize = observe.observe(self.normalize)
        lib.AudioBuffer_clip(buf._obj, threshold, soft, normalize)
        return buf


class ClipByRatio(Base):
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
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     ClipByRatio(0.25)(buf)
        array([[1., 2., 3., 3.]], dtype=float32)

    """
    def __init__(
            self,
            ratio: Union[float, observe.Base],
            *,
            soft: Union[bool, observe.Base] = False,
            normalize: Union[bool, observe.Base] = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.ratio = ratio
        self.soft = soft
        self.normalize = normalize

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        ratio = observe.observe(self.ratio)
        soft = observe.observe(self.soft)
        normalize = observe.observe(self.normalize)
        lib.AudioBuffer_clipByRatio(buf._obj, ratio, soft, normalize)
        return buf


class Compose(Base):
    r"""Compose several transforms together.

    Args:
        transforms: list of transforms to compose
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([0.5, -0.5, 0.5, -0.5], 8000) as buf:
        ...     Compose([GainStage(12), Clip()])(buf)
        array([[ 1., -1.,  1., -1.]], dtype=float32)

    """
    def __init__(
            self,
            transforms: Sequence[Base],
            *,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.transforms = transforms

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        for t in self.transforms:
            t(buf)
        return buf


class CompressDynamicRange(Base):
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
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([0.5, 0.5], 8000) as buf:
        ...     CompressDynamicRange(-6, 0.5)(buf)
        array([[0.56034607,  0.56034607]], dtype=float32)

    """
    def __init__(
            self,
            threshold_db: Union[float, observe.Base],
            ratio: Union[float, observe.Base], *,
            attack_time: Union[float, observe.Base] = 0.01,
            release_time: Union[float, observe.Base] = 0.02,
            knee_radius_db: Union[float, observe.Base] = 4.0,
            makeup_db: Union[None, float, observe.Base] = 0.0,
            clip: Union[bool, observe.Base] = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_time = attack_time
        self.release_time = release_time
        self.knee_radius_db = knee_radius_db
        self.makeup_db = makeup_db
        self.clip = clip

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        threshold_db = observe.observe(self.threshold_db)
        ratio = observe.observe(self.ratio)
        attack_time = observe.observe(self.attack_time)
        release_time = observe.observe(self.release_time)
        knee_radius_db = observe.observe(self.knee_radius_db)
        if self.makeup_db is None:
            makeup_db = np.nan
        else:
            makeup_db = observe.observe(self.makeup_db)
        clip = observe.observe(self.clip)
        lib.AudioBuffer_compressDynamicRange(buf._obj,
                                             threshold_db,
                                             ratio,
                                             attack_time,
                                             release_time,
                                             knee_radius_db,
                                             makeup_db,
                                             clip)
        return buf


class FFTConvolve(Base):
    r"""Convolve the audio buffer (base) with another buffer (auxiliary)
    using an impulse response (FFT-based approach).

    Args:
        aux: auxiliary buffer,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a buffer with the same length
            as the base buffer
            containing zeros
        keep_tail: keep the tail of the convolution result (extending
            the length of the buffer), or to cut it out (keeping the
            original length of the input)
        transform: transformation applied to the auxiliary buffer
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     with AudioBuffer.from_array([0, 1], 8000) as aux:
        ...         FFTConvolve(aux)(buf)
        array([[0., 1., 2., 3., 4.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     with AudioBuffer.from_array([0, 1], 8000) as aux:
        ...         FFTConvolve(aux, keep_tail=False)(buf)
        array([[0., 1., 2., 3.]], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, AudioBuffer, Base],
            *,
            keep_tail: Union[bool, observe.Base] = True,
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
            aux=aux,
            transform=transform,
        )
        self.keep_tail = keep_tail

    @buffer_length_can_change_decorator
    def _call(self, base: AudioBuffer, aux: AudioBuffer):
        keep_tail = observe.observe(self.keep_tail)
        lib.AudioBuffer_fftConvolve(base._obj, aux._obj, keep_tail)


class Function(Base):
    r"""Apply a custom function to the buffer.

    The function gets as input a :class:`numpy.ndarray`
    of shape ``(channels, samples)``
    with the content of the audio buffer
    and the sampling rate.
    Additional arguments can be provided with
    the ``function_args`` dictionary.
    Observable arguments
    (e.g. :class:`auglib.IntUni`)
    are automatically evaluated.
    The function must either return ``None``
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
        function_args: dictionary with additional function arguments
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> buf = AudioBuffer.from_array([1, 2, 3, 4], 8000)
        >>> def plus_c(x, sr, c):
        ...     x += c
        >>> Function(plus_c, {'c': 1})(buf)
        array([[2., 3., 4., 5.]], dtype=float32)
        >>> def halve(x, sr):
        ...     return x[:, ::2]
        >>> Function(halve)(buf)
        array([[2., 4.]], dtype=float32)
        >>> Function(lambda x, sr: x * 2)(buf)
        array([[4., 8.]], dtype=float32)
        >>> buf.free()

    """
    @audobject.init_decorator(
        resolvers={
            'function': audobject.resolver.Function,
        },
    )
    def __init__(
            self,
            function: Callable[..., Optional[np.ndarray]],
            function_args: typing.Dict = None,
            *,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.function = function
        self.function_args = function_args

    def _call(self, buf: AudioBuffer):

        # evaluate function arguments
        args = {}
        if self.function_args:
            for key, value in self.function_args.items():
                args[key] = observe.observe(value)

        # apply function
        x = buf.to_array(copy=False)
        y = self.function(x, buf.sampling_rate, **args)

        # if a new array is returned
        # we copy the result to the buffer,
        # otherwise we assume an inplace operation
        if y is not None:
            # we don't support empty buffers
            if y.size == 0:
                raise RuntimeError(
                    'Buffers must be non-empty. '
                    'Yours is empty '
                    'after applying the following transform: '
                    f"'{str(self)}'."
                )
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
            buf._data[:] = y

        return buf


class GainStage(Base):
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
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     GainStage(to_db(2))(buf)
        array([[2., 4., 6., 8.]], dtype=float32)

    """
    def __init__(
            self,
            gain_db: Union[float, observe.Base],
            *,
            max_peak_db: Union[float, observe.Base] = None,
            clip: Union[bool, observe.Base] = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.gain_db = gain_db
        self.max_peak_db = max_peak_db
        self.clip = clip

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        gain_db = observe.observe(self.gain_db)
        clip = observe.observe(self.clip)
        max_peak_db = observe.observe(self.max_peak_db)
        if max_peak_db is not None:
            lib.AudioBuffer_gainStageSafe(buf._obj, gain_db, max_peak_db)
            if self.clip:
                lib.AudioBuffer_clip(buf._obj, 0.0, False, False)
        else:
            lib.AudioBuffer_gainStage(buf._obj, gain_db, clip)
        return buf


class HighPass(Base):
    r"""Run audio buffer through a high-pass filter.

    Args:
        cutoff: cutoff frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Example:
        >>> with AudioBuffer.from_array([1, 2], 8000) as buf:
        ...     HighPass(3500, order=4)(buf)
        array([[ 0.0009335 , -0.00464588]], dtype=float32)

    """
    def __init__(
            self,
            cutoff: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.cutoff = cutoff
        self.order = order
        if design not in SUPPORTED_FILTER_DESIGNS:
            raise ValueError(
                f"Unknown filter design '{design}'. "
                "Supported designs are: "
                f"{', '.join(SUPPORTED_FILTER_DESIGNS)}."
            )
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        cutoff = observe.observe(self.cutoff)
        order = observe.observe(self.order)
        if self.design == 'butter':
            lib.AudioBuffer_butterworthHighPassFilter(buf._obj, cutoff, order)
        return buf


class LowPass(Base):
    r"""Run audio buffer through a low-pass filter.

    Args:
        cutoff: cutoff frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Example:
        >>> with AudioBuffer.from_array([1, 2], 8000) as buf:
        ...     LowPass(100)(buf)
        array([[0.03780475, 0.14836061]], dtype=float32)

    """
    def __init__(
            self,
            cutoff: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.cutoff = cutoff
        self.order = order
        if design not in SUPPORTED_FILTER_DESIGNS:
            raise ValueError(
                f"Unknown filter design '{design}'. "
                "Supported designs are: "
                f"{', '.join(SUPPORTED_FILTER_DESIGNS)}."
            )
        self.design = design

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        cutoff = observe.observe(self.cutoff)
        order = observe.observe(self.order)
        if self.design == 'butter':
            lib.AudioBuffer_butterworthLowPassFilter(buf._obj, cutoff, order)
        return buf


class Mask(Base):
    r"""Masked transformation.

    Usually a transformation is applied on the whole buffer.
    With :class:`auglib.transform.Mask` it is possible
    to mask the transformation within specific region(s).
    By default,
    regions outside the mask are augmented.
    If ``invert`` is set to ``True``,
    regions that fall inside the mask are augmented.

    Args:
        transform: transform object
        start_pos: start masking at this point (see ``unit``).
        duration: apply masking for this duration (see ``unit``). If
            set to ``None`` masking is applied until the end of the buffer
        step: if not ``None``,
            alternate between masked and non-masked regions
            by the given step duration.
            If two steps are given,
            the first value defines the length of masked regions,
            and the second value the steps between masked regions
            (see ``unit``)
        invert: if set to ``True`` augment the masked regions
        unit: literal specifying the format of ``step``,
            ``start_pos`` and ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 1, 1, 1, 1, 1], 8000) as buf:
        ...     Mask(
        ...         GainStage(to_db(2)),
        ...         start_pos=2,
        ...         duration=3,
        ...         unit='samples',
        ...     )(buf)
        array([[2., 2., 1., 1., 1., 2.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 1, 1, 1, 1, 1], 8000) as buf:
        ...     Mask(
        ...         GainStage(to_db(2)),
        ...         start_pos=2,
        ...         duration=3,
        ...         unit='samples',
        ...         invert=True,
        ...     )(buf)
        array([[1., 1., 2., 2., 2., 1.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 1, 1, 1, 1, 1], 8000) as buf:
        ...     Mask(
        ...         GainStage(to_db(2)),
        ...         step=2,
        ...         unit='samples',
        ...     )(buf)
        array([[1., 1., 2., 2., 1., 1.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 1, 1, 1, 1, 1], 8000) as buf:
        ...     Mask(
        ...         GainStage(to_db(2)),
        ...         step=(2, 1),
        ...         unit='samples',
        ...     )(buf)
        array([[1., 1., 2., 1., 1., 2.]], dtype=float32)

    """  # noqa: E501
    @audobject.init_decorator(
        resolvers={
            'step': audobject.resolver.Tuple,
        }
    )
    def __init__(
            self,
            transform: Base,
            *,
            start_pos: typing.Union[int, float, observe.Base, Time] = 0,
            duration: typing.Union[int, float, observe.Base, Time] = None,
            step: typing.Union[
                int, float, observe.Base, Time,
                typing.Tuple[
                    Union[int, float, observe.Base, Time],
                    Union[int, float, observe.Base, Time],
                ],
            ] = None,
            invert: typing.Union[bool, observe.Base] = False,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: typing.Union[float, observe.Base] = None,
    ):
        if step is not None:
            step = audeer.to_list(step)
            if len(step) == 1:
                step = step * 2
            step = tuple(step)

        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.transform = transform
        self.start_pos = start_pos
        self.duration = duration
        self.step = step
        self.invert = invert
        self.unit = unit

    def _call(self, buf: AudioBuffer):

        # store original signal, then apply transformation
        org_signal = buf.to_array(copy=True).squeeze()
        self.transform(buf)
        aug_signal = buf.to_array(copy=False).squeeze()
        num_samples = min(org_signal.size, aug_signal.size)

        # figure start and end position of mask
        start_pos = to_samples(
            self.start_pos,
            length=num_samples,
            sampling_rate=buf.sampling_rate,
            unit=self.unit,
        )
        if self.duration is None:
            end_pos = num_samples
        else:
            end_pos = start_pos + to_samples(
                self.duration,
                length=num_samples,
                sampling_rate=buf.sampling_rate,
                unit=self.unit,
            )
            end_pos = min(end_pos, num_samples)

        # create mask
        mask = np.zeros(num_samples, dtype=bool)
        mask[:start_pos] = True
        mask[end_pos:] = True

        # apply steps
        if self.step is not None:
            masked_region = True
            while start_pos < end_pos:
                # switch between the two frequencies
                # freq[0] -> mask
                # freq[1] -> no mask
                # and calculate next step
                if masked_region:
                    step = self.step[0]
                else:
                    step = self.step[1]
                step = to_samples(
                    step,
                    length=num_samples,
                    sampling_rate=buf.sampling_rate,
                    unit=self.unit,
                )
                step = min(step, end_pos - start_pos)
                # if we are not in a masked region, revert changes
                if not masked_region and step > 0:
                    mask[start_pos:start_pos + step] = True
                # increment position and switch condition
                start_pos += step
                masked_region = not masked_region

        # apply mask
        # invert = False -> remove augmentation within mask
        # invert = True -> keep augmentation within mask
        if not observe.observe(self.invert):
            mask = ~mask
        aug_signal[:num_samples][mask] = org_signal[:num_samples][mask]

        return buf


class Mix(Base):
    r"""Mix two audio buffers.

    Mix a base and auxiliary buffer
    which may differ in length,
    but must have the same sampling rate.
    Individual gains can be set for both signals
    (``gain_base_db`` and ``gain_aux_db``).
    If ``snr_db`` is specified,
    ``gain_aux_db`` is automatically calculated
    to match the requested signal-to-noise ratio.
    The signal-to-noise ratio
    refers only to the overlapping parts
    of the base and auxiliary buffer.

    ``write_pos_base`` specifies the starting point
    of adding the auxiliary signal
    to the the base buffer.
    Selecting a sub-segment of the auxiliary buffer is possible
    with selecting a starting point (``read_pos_aux``)
    and/or specifying its duration (``read_dur_aux``).

    In order to allow the looping of the auxiliary buffer
    or its selected sub-segment,
    the ``loop_aux`` argument can be used.
    In case the auxiliary buffer ends beyond
    the original ending point,
    the extra portion will be discarded,
    unless ``extend_base`` is set,
    in which case the base buffer is extended accordingly.

    By default,
    the auxiliary buffer is mixed
    into the base buffer exactly once.
    However, the number of repetitions
    can be controlled with ``num_repeat``.
    Usually,
    this only makes sense
    when reading from random positions
    or random files.

    Args:
        aux: auxiliary buffer,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a buffer with the same length
            as the base buffer
            containing zeros
        gain_base_db: gain of base buffer
        gain_aux_db: gain of auxiliary buffer.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise (base-to-aux) ratio in decibels
        write_pos_base: write position of base buffer (see ``unit``)
        read_pos_aux: read position of auxiliary buffer (see ``unit``)
        read_dur_aux: duration to read from auxiliary buffer
            (see ``unit``).
            Set to ``None`` or ``0`` to read the whole buffer
        clip_mix: clip amplitude values of mixed signal to [-1, 1]
        loop_aux: loop auxiliary buffer if shorter than base buffer
        extend_base: if needed, extend base buffer to total required
            length (considering length of auxiliary buffer)
        num_repeat: number of repetitions
        unit: literal specifying the format of ``write_pos_base``,
            ``read_pos_aux`` and ``read_dur_aux``
            (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary buffer
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     with AudioBuffer.from_array([1, 1], 8000) as aux:
        ...         Mix(aux, num_repeat=2)(buf)
        array([[3., 4., 3., 4.]], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, AudioBuffer, Base],
            *,
            gain_base_db: Union[float, observe.Base] = 0.0,
            gain_aux_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            write_pos_base: Union[int, float, observe.Base, Time] = 0.0,
            read_pos_aux: Union[int, float, observe.Base, Time] = 0.0,
            read_dur_aux: Union[int, float, observe.Base, Time] = None,
            clip_mix: Union[bool, observe.Base] = False,
            loop_aux: Union[bool, observe.Base] = False,
            extend_base: Union[bool, observe.Base] = False,
            num_repeat: Union[int, observe.Base] = 1,
            unit='seconds',
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):

        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
            aux=aux,
            transform=transform,
            num_repeat=num_repeat,
        )
        self.gain_base_db = gain_base_db
        self.gain_aux_db = gain_aux_db
        self.snr_db = snr_db
        self.write_pos_base = write_pos_base
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux or 0
        self.clip_mix = clip_mix
        self.loop_aux = loop_aux
        self.extend_base = extend_base
        self.unit = unit

    @buffer_length_can_change_decorator
    def _call(self, base: AudioBuffer, aux: AudioBuffer) -> AudioBuffer:
        write_pos_base = to_samples(
            self.write_pos_base,
            sampling_rate=base.sampling_rate,
            unit=self.unit,
            length=len(base),
        )
        read_pos_aux = to_samples(
            self.read_pos_aux,
            sampling_rate=aux.sampling_rate,
            unit=self.unit,
            length=len(aux),
        )
        read_dur_aux = to_samples(
            self.read_dur_aux,
            sampling_rate=aux.sampling_rate,
            unit=self.unit,
            length=len(aux),
        )
        gain_base_db = observe.observe(self.gain_base_db)
        clip_mix = observe.observe(self.clip_mix)
        loop_aux = observe.observe(self.loop_aux)
        extend_base = observe.observe(self.extend_base)
        if self.snr_db is not None:
            snr_db = observe.observe(self.snr_db)
            # Estimate gain by considering only overlapping parts
            # of the buffers
            len_base = len(base) - write_pos_base
            if read_dur_aux == 0:
                len_aux = len(aux) - read_pos_aux
            else:
                len_aux = min(
                    len(aux) - read_pos_aux,
                    read_dur_aux,
                )
            if len_base > len_aux and not loop_aux:
                len_base = len_aux
            elif len_aux > len_base and not extend_base:
                len_aux = len_base
            rms_base_db = (
                gain_base_db
                + rms_db(base._data[write_pos_base:len_base])
            )
            rms_aux_db = rms_db(
                aux._data[read_pos_aux:read_pos_aux + len_aux]
            )
            gain_aux_db = get_noise_gain_from_requested_snr(
                rms_base_db,
                rms_aux_db,
                snr_db,
            )
        else:
            gain_aux_db = observe.observe(self.gain_aux_db)
        lib.AudioBuffer_mix(
            base._obj,
            aux._obj,
            gain_base_db,
            gain_aux_db,
            write_pos_base,
            read_pos_aux,
            read_dur_aux,
            clip_mix,
            loop_aux,
            extend_base,
        )

        return base


class NormalizeByPeak(Base):
    r"""Peak-normalize the audio buffer to a desired level.

    Args:
        peak_db: desired peak value in decibels
        clip: clip sample values to the interval [-1.0, 1.0]
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([0.5, -0.5, 0.5, -0.5], 8000) as buf:
        ...     NormalizeByPeak()(buf)
        array([[ 1., -1.,  1., -1.]], dtype=float32)
        >>> with AudioBuffer.from_array([0.5, -0.5, 0.5, -0.5], 8000) as buf:
        ...     NormalizeByPeak(peak_db=-3)(buf)
        array([[ 0.7079458, -0.7079458, 0.7079458, -0.7079458]], dtype=float32)

    """
    def __init__(
            self,
            *,
            peak_db: Union[float, observe.Base] = 0.0,
            clip: Union[bool, observe.Base] = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.peak_db = peak_db
        self.clip = clip

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        peak_db = observe.observe(self.peak_db)
        clip = observe.observe(self.clip)
        lib.AudioBuffer_normalizeByPeak(buf._obj, peak_db, clip)
        return buf


class PinkNoise(Base):
    r"""Adds pink noise.

    Args:
        gain_db: gain in decibels
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> seed(0)
        >>> with AudioBuffer.from_array([0, 0], 8000) as buf:
        ...     PinkNoise()(buf)
        array([[0.06546371, 0.06188003]], dtype=float32)

    """
    def __init__(
            self,
            *,
            gain_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.gain_db = gain_db
        self.snr_db = snr_db

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        if self.snr_db is not None:
            snr_db = observe.observe(self.snr_db)
            # The RMS value of pink noise signals
            # generated with a fixed gain
            # has a larger fluctuation
            # than for white noise (> 1dB).
            # To provide a better result
            # for the requested SNR value
            # we measure explicitly the actual RMS
            # of the generated pink noise vector
            # and adjust the gain afterwards.
            with AudioBuffer(
                    len(buf),
                    buf.sampling_rate,
                    unit='samples',
            ) as tmp:
                lib.AudioBuffer_addPinkNoise(tmp._obj, 0)
                rms_noise_db = rms_db(tmp._data)
                rms_signal_db = rms_db(buf._data)
                gain_db = get_noise_gain_from_requested_snr(
                    rms_signal_db,
                    rms_noise_db,
                    snr_db,
                )
                lib.AudioBuffer_mix(
                    buf._obj, tmp._obj, 0, gain_db, 0, 0, 0, False, False,
                    False
                )
        else:
            gain_db = observe.observe(self.gain_db)
            lib.AudioBuffer_addPinkNoise(buf._obj, gain_db)
        return buf


class Prepend(Base):
    r"""Prepend an auxiliary buffer to the base buffer.

    Base and auxiliary buffer may differ in length
    but must have the same sampling rate.

    Args:
        aux: auxiliary buffer,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a buffer with the same length
            as the base buffer
            containing zeros
        read_pos_aux: read position of auxiliary buffer
            (see ``unit``)
        read_dur_aux: duration to read from auxiliary buffer
            (see ``unit``).
            Set to ``None`` or ``0`` to read the whole buffer
        unit: literal specifying the format
            of ``read_pos_aux`` and ``read_dur_aux``
            (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary buffer
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     with AudioBuffer.from_array([5, 6], 8000) as aux:
        ...         Prepend(aux)(buf)
        array([[5., 6., 1., 2., 3., 4.]], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, AudioBuffer, Base],
            *,
            read_pos_aux: Union[int, float, observe.Base, Time] = 0.0,
            read_dur_aux: Union[int, float, observe.Base, Time] = None,
            unit: str = 'seconds',
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.aux = aux
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux or 0
        self.unit = unit
        self.transform = transform

    @buffer_length_can_change_decorator
    def _call(self, base: AudioBuffer, aux: AudioBuffer) -> AudioBuffer:
        read_pos_aux = to_samples(
            self.read_pos_aux,
            sampling_rate=aux.sampling_rate,
            unit=self.unit,
        )
        read_dur_aux = to_samples(
            self.read_dur_aux,
            sampling_rate=aux.sampling_rate,
            unit=self.unit,
        )
        lib.AudioBuffer_prepend(
            base._obj,
            aux._obj,
            read_pos_aux,
            read_dur_aux,
        )
        return base


class PrependValue(Base):
    r"""Prepend base buffer with a constant value.

    Args:
        duration: duration of buffer with constant value
            that will be prepended
            (see ``unit``)
        value: value to prepend
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     PrependValue(2, 5, unit='samples')(buf)
        array([[5., 5., 1., 2., 3., 4.]], dtype=float32)

    """
    def __init__(
            self,
            duration: Union[int, float, observe.Base, Time],
            value: Union[float, observe.Base] = 0,
            *,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.duration = duration
        self.value = value
        self.unit = unit

    @buffer_length_can_change_decorator
    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        if self.duration != 0:
            with AudioBuffer(
                    duration=self.duration,
                    sampling_rate=buf.sampling_rate,
                    value=self.value,
                    unit=self.unit,
            ) as aux:
                Prepend(aux)(buf)
        return buf


class Resample(Base):
    r"""Resample signal to another sampling rate.

    Changes number of samples of the audio buffer
    by applying :func:`audresample.resample` to the signal.
    By default,
    this will not change the sampling rate of the audio buffer.
    However,
    if ``override=True`` the sampling rate
    of the audio buffer will be set to ``target_rate``.
    This is useful to provide a desired input sampling rate
    for other transforms such as :class:`auglib.transform.AMRNB`.
    But the sampling rate of augmented signals
    generated with :class:`auglib.Augment` is never affected.

    Args:
        target_rate: target rate in Hz
        override: override sampling rate of buffer with target rate
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([0, 0, 0, 0], 16000) as buf:
        ...     Resample(8000)(buf)
        array([[0., 0.]], dtype=float32)

    """
    def __init__(
            self,
            target_rate: typing.Union[int, observe.List],
            *,
            override: bool = False,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: typing.Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.target_rate = target_rate
        self.override = override

    def _call(self, buf: AudioBuffer) -> AudioBuffer:

        original_rate = buf.sampling_rate
        target_rate = observe.observe(self.target_rate)

        if original_rate != target_rate:

            # resample buffer
            x = buf.to_array(copy=False)
            y = audresample.resample(
                x,
                original_rate,
                target_rate,
            )

            # fit buffer size to result
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
            buf._data[:] = y

            # possibly change sampling rate of buffer
            if self.override:
                buf.sampling_rate = target_rate
                lib.AudioBuffer_setSampleRate(buf._obj, target_rate)

        return buf


class Select(Base):
    r"""Randomly select from a pool of transforms.

    Args:
        transforms: list of transforms to choose from
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> seed(0)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     with AudioBuffer.from_array([0., 0.], 8000) as aux:
        ...         Select([Prepend(aux), Append(aux)])(buf)
        array([[0., 0., 1., 2., 3., 4.]], dtype=float32)

    """
    def __init__(
            self,
            transforms: Sequence[Base],
            *,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.transforms = transforms

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        idx = np.random.randint(len(self.transforms))
        self.transforms[idx](buf)
        return buf


class Shift(Base):
    r"""Shift signal without changing its duration.

    The buffer will be read
    from the position
    given by ``duration``,
    and the skipped samples from the beginng
    will be added at the end.

    Args:
        duration: duration of shift
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Shift(1, unit='samples')(buf)
        array([[2., 3., 4., 1.]], dtype=float32)

    """
    def __init__(
            self,
            duration: typing.Union[int, float, observe.Base, Time] = None,
            *,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: typing.Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.duration = duration
        self.unit = unit

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        if self.duration is not None:
            duration = to_samples(
                observe.observe(self.duration),
                sampling_rate=buf.sampling_rate,
                unit=self.unit,
                length=len(buf),
            )
            # Allow shift values that are out-of-bound
            # of the actual buffer duration
            start_pos = duration % len(buf)
            if start_pos == 0:
                return buf
            # Append data
            # that will be removed at the beginning
            with AudioBuffer.from_array(
                buf._data[:start_pos],
                buf.sampling_rate,
            ) as aux:
                Append(aux)(buf)
            # Remove data from beginning
            Trim(start_pos=start_pos, unit='samples')(buf)

        return buf


class Tone(Base):
    r"""Adds basic waveform.

    The sine waveform will start at 0,
    the square and sawtooth waveform at -1,
    and the triangle waveform at 1.
    The waveform sawtooth has a rising ramp.

    Args:
        freq: fundamental frequency in Hz
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        shape: tone shape,
            one of ``'sine'``,
            ``'square'``,
            ``'triangle'``,
            ``'sawtooth'``
        lfo_rate: modulation rate of Low Frequency Oscillator
        lfo_range: modulation range of Low Frequency Oscillator
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``shape`` contains a non-supported value

    Example:
        >>> with AudioBuffer.from_array([0, 0, 0, 0], 8000) as buf:
        ...     Tone(2000, shape='sine')(buf)
        array([[ 0.000000e+00,  1.000000e+00, -8.742278e-08, -1.000000e+00]],
              dtype=float32)
        >>> with AudioBuffer.from_array([0, 0, 0, 0], 8000) as buf:
        ...     Tone(2000, shape='square')(buf)
        array([[-1., -1.,  0.,  1.]], dtype=float32)
        >>> with AudioBuffer.from_array([0, 0, 0, 0], 8000) as buf:
        ...     Tone(2000, shape='sawtooth')(buf)
        array([[-1. , -0.5,  0. ,  0.5]], dtype=float32)
        >>> with AudioBuffer.from_array([0, 0, 0, 0], 8000) as buf:
        ...     Tone(2000, shape='triangle')(buf)
        array([[ 1.,  0., -1.,  0.]], dtype=float32)

    """  # noqa: E501
    def __init__(
            self,
            freq: Union[float, observe.Base],
            *,
            gain_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            shape: str = 'sine',
            lfo_rate: Union[float, observe.Base] = 0.0,
            lfo_range: Union[float, observe.Base] = 0.0,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.freq = freq
        self.gain_db = gain_db
        self.snr_db = snr_db
        self.lfo_rate = lfo_rate
        self.lfo_range = lfo_range
        if shape not in SUPPORTED_TONE_SHAPES:
            raise ValueError(
                f"Unknown tone shape '{shape}'. "
                "Supported shapes are: "
                f"{', '.join(SUPPORTED_TONE_SHAPES)}."
            )
        self.shape = shape

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        freq = observe.observe(self.freq)
        lfo_rate = observe.observe(self.lfo_rate)
        lfo_range = observe.observe(self.lfo_range)
        if self.snr_db is not None:
            snr_db = observe.observe(self.snr_db)
            # RMS values of the tone,
            # see https://en.wikipedia.org/wiki/Root_mean_square
            #
            # For short signals and low sampling rates,
            # rms_tone_db and hence the resulting SNR
            # can slightly deviate from the theoretical value.
            if self.shape == 'sine':
                rms = 1 / np.sqrt(2)
            elif self.shape == 'square':
                rms = 1
            elif self.shape == 'triangle':
                rms = 1 / np.sqrt(3)
            elif self.shape == 'sawtooth':
                rms = 1 / np.sqrt(3)
            rms_tone_db = 20 * np.log10(rms)
            rms_signal_db = rms_db(buf._data)
            gain_db = get_noise_gain_from_requested_snr(
                rms_signal_db,
                rms_tone_db,
                snr_db,
            )
        else:
            gain_db = observe.observe(self.gain_db)
        if self.shape == 'sine':
            shape_value = 0
        elif self.shape == 'square':
            shape_value = 1
        elif self.shape == 'triangle':
            shape_value = 2
        elif self.shape == 'sawtooth':
            shape_value = 3
        lib.AudioBuffer_addTone(
            buf._obj,
            freq,
            gain_db,
            shape_value,
            lfo_rate,
            lfo_range,
        )
        return buf


class Trim(Base):
    r"""Trim, zero pad, and/or repeat base buffer.

    If ``duration`` is ``None``
    the signal will be trimmed
    to the range [``start_pos``, ``end_pos``],
    whereas the start or end of the signal is used
    if ``start_pos`` and/or ``end_pos`` are ``None``.

    If ``duration`` is provided
    and ``fill`` is ``'none'``
    it will calculate ``start_pos`` or ``end_pos``
    to match the given duration
    if the incoming signal is long enough.
    If ``duration`` is provided,
    but neither ``start_pos`` or ``end_pos``
    it will trim a signal with given ``duration``
    from the center of the signal.

    If ``duration`` is provided
    and ``fill`` is ``'zeros'`` or ``'loop'``
    it will return a signal of length ``duration``
    filling missing values with zeros
    or the trimmed signal.
    ``fill_pos`` defines if the signal is filled
    on the right, left, or both ends.

    The following table shows
    a few combinations of the arguments
    and the resulting augmented signal
    for an ingoing signal of ``[1, 2, 3, 4]``.
    All time values are given in samples.

    ========= ======= ======== ===== ======== ============
    start_pos end_pos duration fill  fill_pos outcome
    ========= ======= ======== ===== ======== ============
    1         None    None     none  right    [2, 3, 4]
    None      1       None     none  right    [1, 2, 3]
    None      None    2        none  right    [2, 3]
    2         None    4        loop  right    [3, 4, 3, 4]
    1         1       4        zeros right    [2, 3, 0, 0]
    3         None    4        zeros both     [0, 4, 0, 0]
    ========= ======= ======== ===== ======== ============

    Args:
        start_pos: starting point of the trimmed region,
            relative to the start of the input buffer
            (see ``unit``)
        end_pos: end point of the trimmed region,
            relative to the end of the input buffer
            (see ``unit``).
            The end point is counted backwards
            from the end of the signal.
            If ``end_pos=512`` samples,
            it will remove the last 512 samples of the input
        duration: target duration of the resulting buffer
            (see ``unit``).
            If set to ``None`` or ``0``,
            the selected section extends
            until the end
            or the beginning
            of the original buffer
        fill: fill strategy
            if the end and/or start point
            of the trimmed region
            exceeds the buffer.
            Three strategies are available:
            ``'none'`` the signal is not extended,
            ``'zeros'`` the signal is filled up with zeros,
            ``'loop'`` the trimmed signal is repeated
        fill_pos: position at which the selected fill strategy applies.
            ``'right'`` adds samples to the right,
            ``'left'`` adds samples to the left,
            or ``'both'`` adds samples on both sides,
            equally distributed starting at the right
        unit: literal specifying the format
            of ``start_pos``,
            ``end_pos``,
            and ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: Probability to bypass the transformation

    Raises:
        ValueError: if ``fill`` contains a non-supported value
        ValueError: if ``fill_pos`` contains a non-supported value

    Example:
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Trim(start_pos=1, unit='samples')(buf)
        array([[2., 3., 4.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Trim(end_pos=1, unit='samples')(buf)
        array([[1., 2., 3.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Trim(start_pos=None, duration=2, unit='samples')(buf)
        array([[2., 3.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Trim(start_pos=2, duration=4, unit='samples', fill='loop')(buf)
        array([[3., 4., 3., 4.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Trim(
        ...         start_pos=1,
        ...         end_pos=1,
        ...         duration=4,
        ...         unit='samples',
        ...         fill='zeros',
        ...     )(buf)
        array([[2., 3., 0., 0.]], dtype=float32)
        >>> with AudioBuffer.from_array([1, 2, 3, 4], 8000) as buf:
        ...     Trim(
        ...         start_pos=3,
        ...         duration=4,
        ...         unit='samples',
        ...         fill='zeros',
        ...         fill_pos='both',
        ...     )(buf)
        array([[0., 4., 0., 0.]], dtype=float32)

    """
    def __init__(
            self,
            *,
            start_pos: Union[int, float, observe.Base, Time] = 0,
            end_pos: Union[int, float, observe.Base, Time] = None,
            duration: Union[int, float, observe.Base, Time] = None,
            fill: str = 'none',
            fill_pos: str = 'right',
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        if fill not in SUPPORTED_FILL_STRATEGIES:
            raise ValueError(
                f"Unknown fill strategy '{fill}'. "
                "Supported strategies are: "
                f"{', '.join(SUPPORTED_FILL_STRATEGIES)}."
            )
        if fill_pos not in SUPPORTED_FILL_POSITIONS:
            raise ValueError(
                f"Unknown fill_pos '{fill_pos}'. "
                "Supported positions are: "
                f"{', '.join(SUPPORTED_FILL_POSITIONS)}."
            )
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.fill = fill
        self.fill_pos = fill_pos
        self.unit = unit

    @buffer_length_can_change_decorator
    def _call(self, buf: AudioBuffer) -> AudioBuffer:

        # start_pos | end_pos | duration
        # --------- | ------- | --------
        # None      | None    | None/0
        #
        # Return signal without trimming
        if (
                self.start_pos is None
                and self.end_pos is None
                and (
                    self.duration is None
                    or self.duration == 0
                )
        ):
            return buf

        # Convert start_pos, end_pos, duration to samples
        # and check for meaningful values
        length = len(buf)
        if self.start_pos is not None:
            start_pos = to_samples(
                observe.observe(self.start_pos),
                sampling_rate=buf.sampling_rate,
                unit=self.unit,
                length=length,
                allow_negative=True,
            )
            if start_pos >= length:
                raise ValueError(f"'start_pos' must be <{length}.")
            if start_pos < 0:
                raise ValueError("'start_pos' must be >=0.")
        if self.end_pos is not None:
            end_pos = to_samples(
                observe.observe(self.end_pos),
                sampling_rate=buf.sampling_rate,
                unit=self.unit,
                length=length,
                allow_negative=True,
            )
            if end_pos >= length:
                raise ValueError(f"'end_pos' must be <{length}.")
            if end_pos < 0:
                raise ValueError("'end_pos' must be >=0.")
        if self.duration is not None:
            if self.duration == 0:
                duration = length
            else:
                duration = to_samples(
                    observe.observe(self.duration),
                    sampling_rate=buf.sampling_rate,
                    unit=self.unit,
                    length=length,
                    allow_negative=True,
                )
            if duration < 0:
                raise ValueError("'duration' must be >=0.")
            # duration can become 0 if self.duration is very small
            if duration == 0:
                raise ValueError(
                    "Your combination of "
                    f"'duration' = {self.duration} {self.unit} "
                    f"and 'sampling_rate' = {buf.sampling_rate} Hz "
                    "would lead to an empty buffer "
                    "which is forbidden."
                )

        # start_pos | end_pos | duration
        # --------- | ------- | --------
        # value     | value   |
        #
        # If start_pos and end_pos are given
        # we need to ensure
        # that they would not result
        # in an empty array
        if (
                self.start_pos is not None
                and self.end_pos is not None
        ):
            if length - start_pos - end_pos <= 0:
                raise ValueError(f"'start_pos' + 'end_pos' must be <{length}.")

        # start_pos | end_pos | duration
        # --------- | ------- | --------
        #           |         | None
        #
        # Calculate duration if not given
        if self.duration is None:
            if self.start_pos is None:
                start_pos = 0
            elif self.end_pos is None:
                end_pos = 0
            duration = len(buf) - start_pos - end_pos

        # start_pos | end_pos | duration
        # --------- | ------- | --------
        #           |         | value
        #
        # Return trimmed signal based on duration.
        # If the duration is longer
        # than the incoming signal
        # or the trimmed signal
        # based on start_pos and/or end_pos,
        # the signal is also filled
        # according to fill and fill_pos
        if self.duration is not None:

            def distribute_samples(num_samples):
                r"""Distribute samples to left and right.

                If ``num_samples`` is zero or negative
                ``0, 0`` is returned.

                If ``num_samples`` is even
                more samples are returned on the right side.

                """
                if num_samples <= 0:
                    return 0, 0
                start = int(num_samples / 2)
                end = int(num_samples / 2)
                # For even numbers add one sample to the right
                if num_samples % 2 != 0:
                    end += num_samples
                return start, end

            # First trim to [start_pos, end_pos]
            if (
                    self.start_pos is None
                    and self.end_pos is None
            ):
                # If signal is longer than duration cut from center
                start_pos, end_pos = distribute_samples(length - duration)
            elif self.start_pos is None:
                start_pos = max(0, length - duration - end_pos)
            elif self.end_pos is None:
                end_pos = max(0, length - duration - start_pos)
            Trim(start_pos=start_pos, end_pos=end_pos, unit='samples')(buf)

            # Check the difference in samples
            # between current buffer
            # and desired duration
            difference = len(buf) - duration

            # Expand buffer if too short
            # and fill is requested
            if (
                    difference < 0
                    and self.fill != 'none'
            ):

                if self.fill_pos == 'right':
                    prepend_samples = 0
                    append_samples = -difference
                elif self.fill_pos == 'left':
                    prepend_samples = -difference
                    append_samples = 0
                elif self.fill_pos == 'both':
                    prepend_samples, append_samples = distribute_samples(
                        -difference
                    )

                if self.fill == 'zeros':
                    # Expand buffer by zeros
                    prepend_array = np.zeros(prepend_samples)
                    append_array = np.zeros(append_samples)
                elif self.fill == 'loop':
                    # Repeat signal in the expanded parts
                    repetitions = (
                        int(max(prepend_samples, append_samples) / len(buf))
                        + 1
                    )
                    repeated_array = np.tile(buf._data, repetitions)
                    prepend_array = repeated_array[-prepend_samples:]
                    append_array = repeated_array[:append_samples]

                if prepend_samples > 0:
                    with AudioBuffer.from_array(
                        prepend_array,
                        buf.sampling_rate,
                    ) as aux:
                        Prepend(aux)(buf)
                if append_samples > 0:
                    with AudioBuffer.from_array(
                        append_array,
                        buf.sampling_rate,
                    ) as aux:
                        Append(aux)(buf)

            # Set start_pos to 0 for final trim with provided duration
            start_pos = 0

        lib.AudioBuffer_trim(buf._obj, start_pos, duration)

        return buf


class WhiteNoiseGaussian(Base):
    r"""Adds Gaussian white noise.

    Args:
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        stddev: standard deviation
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> seed(0)
        >>> with AudioBuffer.from_array([0, 0], 8000) as buf:
        ...     WhiteNoiseGaussian()(buf)
        array([[-0.12818003,  0.33024564]], dtype=float32)

    """
    def __init__(
            self,
            *,
            gain_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            stddev: Union[float, observe.Base] = 0.3,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.gain_db = gain_db
        self.snr_db = snr_db
        self.stddev = stddev

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        if self.snr_db is not None:
            snr_db = observe.observe(self.snr_db)
            # For white noise we have
            # rms_noise_db
            # = 20 * log10(rms_noise)
            # = 10 * log10(power_noise)
            # = 10 * log10(stddev^2)
            # compare https://en.wikipedia.org/wiki/White_noise
            #
            # For short signals and low sampling rates,
            # rms_noise_db and hence the resulting SNR
            # can slightly deviate from the theoretical value.
            rms_noise_db = 10 * np.log10(self.stddev ** 2)
            rms_signal_db = rms_db(buf._data)
            gain_db = get_noise_gain_from_requested_snr(
                rms_signal_db,
                rms_noise_db,
                snr_db,
            )
        else:
            gain_db = observe.observe(self.gain_db)
        stddev = observe.observe(self.stddev)
        lib.AudioBuffer_addWhiteNoiseGaussian(buf._obj, gain_db, stddev)
        return buf


class WhiteNoiseUniform(Base):
    """Adds uniform white noise.

    Args:
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented buffer
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Example:
        >>> seed(0)
        >>> with AudioBuffer.from_array([0, 0], 8000) as buf:
        ...     WhiteNoiseUniform()(buf)
        array([[-0.796302  ,  0.89579505]], dtype=float32)

    """
    def __init__(
            self,
            *,
            gain_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.gain_db = gain_db
        self.snr_db = snr_db

    def _call(self, buf: AudioBuffer) -> AudioBuffer:
        if self.snr_db is not None:
            snr_db = observe.observe(self.snr_db)
            # For uniform white noise we have
            # rms_noise_db ≈ -4.77125 dB
            # deducted as the mean RMS of multiple
            # randomly initialised Uniform White Noises
            # with 0 dB as gain
            #
            # For short signals and low sampling rates,
            # rms_noise_db and hence the resulting SNR
            # can slightly deviate.
            rms_noise_db = -4.77125
            rms_signal_db = rms_db(buf._data)
            gain_db = get_noise_gain_from_requested_snr(
                rms_signal_db,
                rms_noise_db,
                snr_db,
            )
        else:
            gain_db = observe.observe(self.gain_db)
        lib.AudioBuffer_addWhiteNoiseUniform(buf._obj, gain_db)
        return buf
