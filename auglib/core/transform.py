import math
import tempfile
import typing
from typing import Callable
from typing import Sequence
from typing import Union
import warnings

import numpy as np
import scipy.signal

import audeer
import audiofile
import audmath
import audobject
import audresample

from auglib.core import observe
from auglib.core.resolver import ArrayResolver
from auglib.core.resolver import ObservableListResolver
from auglib.core.seed import get_seed
from auglib.core.time import Time
from auglib.core.utils import from_db
from auglib.core.utils import get_peak
from auglib.core.utils import rms_db
from auglib.core.utils import to_db
from auglib.core.utils import to_samples


DTYPE = 'float32'
SUPPORTED_FADE_SHAPES = [
    'tukey',
    'kaiser',
    'linear',
    'exponential',
    'logarithmic',
]
SUPPORTED_FILL_STRATEGIES = ['none', 'zeros', 'loop']
SUPPORTED_FILL_POSITIONS = ['right', 'left', 'both']
SUPPORTED_FILTER_DESIGNS = ['butter']
SUPPORTED_TONE_SHAPES = ['sine', 'square', 'triangle', 'sawtooth']


def get_noise_gain_from_snr(rms_signal_db, rms_noise_db, snr_db):
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
    r"""Base class for transforms applied to a signal.

    Args:
        unit: unit of any duration values
        sampling_rate: sampling rate in Hz
        bypass_prob: probability to bypass the transformation
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        aux: auxiliary signal,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a signal with the same length
            as the base signal
            containing zeros
        transform: transformation applied to the auxiliary signal
        num_repeat: number of repetitions

    """
    @audobject.init_decorator(
        resolvers={
            'aux': ArrayResolver,
        }
    )
    def __init__(
            self,
            bypass_prob: Union[float, observe.Base] = None,
            *,
            unit: str = 'seconds',
            sampling_rate: int = None,
            preserve_level: Union[bool, observe.Base] = False,
            aux: Union[str, observe.Base, np.ndarray, 'Base'] = None,
            transform: 'Base' = None,
            num_repeat: int = None,
    ):
        self.unit = unit
        self.sampling_rate = sampling_rate
        self.bypass_prob = bypass_prob
        self.preserve_level = preserve_level
        self.aux = aux
        self.transform = transform
        self.num_repeat = num_repeat

    def _call(
            self,
            base: np.ndarray,
            aux: np.ndarray = None,
    ):  # pragma: no cover
        r"""Transform a signal.

        Args:
            base: audio signal
            aux: auxiliary signal

        Raises:
            NotImplementedError: raised if not overwritten in child class

        """
        raise NotImplementedError()

    def __call__(self, base: np.ndarray) -> np.ndarray:

        bypass_prob = observe.observe(self.bypass_prob)
        preserve_level = observe.observe(self.preserve_level)
        if (
                bypass_prob is None
                or np.random.random_sample() >= bypass_prob
        ):
            # (sample) => (channel, samples)
            ndim = base.ndim
            base = np.atleast_2d(base)

            if base.dtype != DTYPE:
                base = base.astype(DTYPE)

            if preserve_level:
                base_level = rms_db(base)

            num_repeat = observe.observe(self.num_repeat) or 1
            for _ in range(num_repeat):
                if self.aux is None:
                    base = self._call(base)
                else:
                    aux = observe.observe(self.aux)

                    # aux is file
                    if isinstance(aux, str):
                        aux, _ = audiofile.read(aux)

                    # aux is signal generating transform
                    elif isinstance(aux, Base):
                        generator = aux
                        channels, samples = base.shape
                        aux = np.zeros((channels, samples), dtype=DTYPE)
                        aux = generator(aux)

                    if self.transform is not None:
                        aux = self.transform(aux)

                    aux = np.atleast_2d(aux)
                    if aux.dtype != DTYPE:
                        aux = aux.astype(DTYPE)
                    base = self._call(base, aux)

            if preserve_level:
                mix_level = rms_db(base)
                gain = from_db(base_level - mix_level)
                base = gain * base

            if ndim == 1 and base.shape[0] > 0 and base.ndim > 1:
                # (channel, sample) => (sample)
                base = base.squeeze(axis=0)

            if base.dtype != DTYPE:
                base = base.astype(DTYPE)

        return base

    def to_samples(
            self,
            value: typing.Union[int, float, observe.Base, Time],
            *,
            length: int = None,
            allow_negative=True,
    ) -> int:
        r"""Convert duration value to samples."""
        return to_samples(
            value,
            sampling_rate=self.sampling_rate,
            unit=self.unit,
            length=length,
            allow_negative=allow_negative,
        )


class AMRNB(Base):
    r"""Encode-decode signal using AMRNB codec.

    Adaptive Multi Rate - Narrow Band (AMRNB) speech codec.
    A lossy format used in 3rd generation mobile telephony
    and defined in 3GPP TS 26.071 et al.

    .. note:: The input signal is treated as narrow-band,
        and a sampling rate of 8000 Hz is assumed.

    .. note:: Supported bit rates:
        4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200.
        Any positive bit rate is allowed,
        but it will be internally converted
        to the closest among those listed above.

    Args:
        bit_rate: target bit rate of the encoded stream (in bits per second)
        dtx: enable discontinuous transmission (DTX)
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([0.5, 0.5, 0.5, 0.5])
        >>> base = AMRNB(7400)(base)
        >>> np.max(base)
        0.017089844

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
        self.sampling_rate = 8000

    def _call(self, base: np.ndarray) -> np.ndarray:
        bit_rate = observe.observe(self.bit_rate)
        dtx = observe.observe(self.dtx)

        # ffmpeg requires libavcodec-extra57 under Ubuntu
        # (which install also libvo-amrwbenc0)
        with tempfile.TemporaryDirectory() as tmp:
            infile = audeer.path(tmp, 'infile.wav')
            outfile = audeer.path(tmp, 'outfile.amr')
            audiofile.write(infile, base, self.sampling_rate)
            cmd = [
                'ffmpeg',
                '-i', infile,
                '-b:a', str(bit_rate),
                '-dtx', str(int(dtx)),
                outfile,
            ]
            audiofile.core.utils.run(cmd)
            base, _ = audiofile.read(
                outfile,
                always_2d=True,
                dtype=str(base.dtype),
            )
        return base


class Append(Base):
    r"""Append an auxiliary signal to the base signal.

    Args:
        aux: auxiliary signal,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a signal with the same length
            as the base signal
            containing zeros
        read_pos_aux: read position of auxiliary signal (see ``unit``)
        read_dur_aux: duration to read from auxiliary signal
            (see ``unit``).
            Set to ``None`` or ``0`` to read the whole signal
        unit: literal specifying the format of ``read_pos_aux`` and
            ``read_dur_aux`` (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary signal
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> aux = np.array([5, 6])
        >>> Append(aux)(base)
        array([1., 2., 3., 4., 5., 6.], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, np.ndarray, Base],
            *,
            read_pos_aux: Union[int, float, observe.Base, Time] = 0.0,
            read_dur_aux: Union[int, float, observe.Base, Time] = None,
            unit: str = 'seconds',
            sampling_rate: int = None,
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            aux=aux,
            sampling_rate=sampling_rate,
            unit=unit,
            transform=transform,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux or 0

    def _call(self, base: np.ndarray, aux: np.ndarray) -> np.ndarray:
        if self.read_pos_aux != 0:
            read_pos_aux = self.to_samples(self.read_pos_aux)
        else:
            read_pos_aux = 0
        if self.read_dur_aux == 0:
            read_dur_aux = aux.shape[1] - read_pos_aux
        else:
            read_dur_aux = self.to_samples(self.read_dur_aux)
        base = np.concatenate(
            [base, aux[:, read_pos_aux:read_pos_aux + read_dur_aux]],
            axis=1,
        )
        return base


class AppendValue(Base):
    r"""Expand signal with a constant value.

    Args:
        duration: duration of signal with constant value
            that will be appended
            (see ``unit``)
        value: value to append
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`).
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> AppendValue(2, 5, unit='samples')(base)
        array([1., 2., 3., 4., 5., 5.], dtype=float32)

    """
    def __init__(
            self,
            duration: Union[int, float, observe.Base, Time],
            value: Union[float, observe.Base] = 0,
            *,
            sampling_rate: int = None,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            unit=unit,
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )
        self.duration = duration
        self.value = value

    def _call(self, base: np.ndarray) -> np.ndarray:
        if self.duration != 0:
            samples = self.to_samples(self.duration)
            aux = np.array([[self.value] * samples])
            base = Append(aux)(base)
        return base


class BabbleNoise(Base):
    """Adds Babble Noise.

    Babble noise refers to having several speakers
    in the background
    all talking at the same time.

    :class:`BabbleNoise` does not use built-in speech signals
    but expects a sequence of speech signals or files
    as ``speech`` argument,
    from which it then randomly samples the speech.

    Args:
        speech: speech signal(s) used to create babble noise
        num_speakers: number of speech signals
            used to create babble noise.
            If not enough speech signals are given
            it will repeat all
            or some of them
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        sampling_rate: sampling rate in Hz
        unit: literal specifying the format of ``write_pos_base``,
            ``read_pos_aux`` and ``read_dur_aux``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> seed(1)
        >>> base = np.array([0, 0, 0, 0])
        >>> speech = np.array([1, 0, 0])
        >>> BabbleNoise([speech], num_speakers=2)(base)
        array([0. , 0.5, 0.5, 0. ], dtype=float32)
    """
    @audobject.init_decorator(
        resolvers={
            'speech': ObservableListResolver,
        }
    )
    def __init__(
            self,
            speech: Sequence[Union[str, np.ndarray]],
            *,
            num_speakers: Union[int, observe.Base] = 5,
            gain_db: Union[float, observe.Base] = 0.0,
            snr_db: Union[float, observe.Base] = None,
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            unit=unit,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.speech = speech
        self.num_speakers = num_speakers
        self.gain_db = gain_db
        self.snr_db = snr_db

    def _call(self, base: np.ndarray) -> np.ndarray:
        # First create signal containing babble noise
        # by summing speech signalss
        babble = np.zeros(base.shape, dtype=DTYPE)
        num_repeat = observe.observe(self.num_speakers)
        transform = Mix(
            observe.List(self.speech, draw=True),
            gain_aux_db=to_db(1 / num_repeat),
            num_repeat=num_repeat,
            loop_aux=True,
            # Cycle the input signal
            transform=Shift(
                observe.FloatUni(0, base.shape[1]),
                unit='samples',
            ),
        )
        babble = transform(babble)
        # Mix the babble noise to aux signal
        transform = Mix(
            babble,
            gain_aux_db=self.gain_db,
            snr_db=self.snr_db,
        )
        base = transform(base)

        return base


class BandPass(Base):
    r"""Run signal through a band-pass filter.

    Args:
        center: center frequency in Hz
        bandwidth: bandwidth frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Examples:
        >>> base = np.array([1, 2])
        >>> BandPass(2000, 1000)(base)
        array([0.16591068, 0.53136045], dtype=float32)

    """
    def __init__(
            self,
            center: Union[float, observe.Base],
            bandwidth: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray) -> np.ndarray:
        center = observe.observe(self.center)
        bandwidth = observe.observe(self.bandwidth)
        order = observe.observe(self.order)
        lowcut = center - bandwidth / 2
        highcut = center + bandwidth / 2
        if self.design == 'butter':
            b, a = scipy.signal.butter(
                order,
                [lowcut, highcut],
                btype='bandpass',
                fs=self.sampling_rate,
            )
            base = scipy.signal.lfilter(b, a, base)
        return base.astype(DTYPE)


class BandStop(Base):
    r"""Run signal through a band-stop filter.

    Args:
        center: center frequency in Hz
        bandwidth: bandwidth frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Examples:
        >>> base = np.array([1, 2])
        >>> BandStop(2000, 1000)(base)
        array([0.83408934, 1.4686396 ], dtype=float32)

    """
    def __init__(
            self,
            center: Union[float, observe.Base],
            bandwidth: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray) -> np.ndarray:
        center = observe.observe(self.center)
        bandwidth = observe.observe(self.bandwidth)
        order = observe.observe(self.order)
        lowcut = center - bandwidth / 2
        highcut = center + bandwidth / 2
        if self.design == 'butter':
            b, a = scipy.signal.butter(
                order,
                [lowcut, highcut],
                btype='bandstop',
                fs=self.sampling_rate,
            )
            base = scipy.signal.lfilter(b, a, base)
        return base.astype(DTYPE)


class Clip(Base):
    r"""Hard/soft-clip the signal.

    ``threshold`` sets the amplitude level in decibels
    to which the signal is clipped.
    The optional argument ``soft``
    triggers a soft-clipping behaviour,
    for which the whole waveform
    is warped through a cubic non-linearity,
    resulting in a smooth transition
    between the flat (clipped) regions
    and the rest of the waveform.

    Args:
        threshold: amplitude level
            above which samples will be clipped
            (in decibels)
        soft: apply soft-clipping
        normalize: after clipping normalize signal to 0 decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, -3, 4])
        >>> Clip(threshold=to_db(2))(base)
        array([ 1.,  2., -2.,  2.], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.threshold = threshold
        self.soft = soft
        self.normalize = normalize

    def _call(self, base: np.ndarray) -> np.ndarray:
        threshold = observe.observe(self.threshold)
        soft = observe.observe(self.soft)
        normalize = observe.observe(self.normalize)

        # Clip
        threshold = from_db(threshold)
        base = np.clip(base, -threshold, threshold)

        # Cubic warping for soft clip
        if soft:
            beta = (1 / threshold) ** 2
            k = 3 * threshold / (3 * threshold - beta * threshold ** 3)
            base = k * (base - beta * (base ** 3 / 3))

        if normalize:
            base = NormalizeByPeak(peak_db=0)(base)

        return base



class ClipByRatio(Base):
    r"""Hard/soft-clip a certain fraction of the signal.

    Rather than receiving a specific amplitude threshold,
    this function is designed to get instructed
    about the number of samples
    that are meant to be clipped,
    in relation to the total length of the signal.
    This ratio is internally translated
    into the amplitude threshold
    needed for achieving the specified intensity
    of the degradation.
    The optional argument ``soft``
    triggers a soft-clipping behaviour,
    for which the whole waveform is warped
    through a cubic non-linearity,
    resulting in a smooth transition
    between the flat (clipped) regions
    and the rest of the waveform.

    Args:
        ratio: ratio between the number of samples that are to be clipped
            and the total number of samples in the signal
        soft: apply soft-clipping
        normalize: after clipping normalize signal to 0 decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> ClipByRatio(0.25)(base).round(4)
        array([1., 2., 3., 3.], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.ratio = ratio
        self.soft = soft
        self.normalize = normalize

    def _call(self, base: np.ndarray) -> np.ndarray:
        ratio = observe.observe(self.ratio)
        soft = observe.observe(self.soft)
        normalize = observe.observe(self.normalize)

        if ratio == 0:
            return base

        # Find threshold by ordering peak levels
        # and selecting based on the ratio
        samples_to_clip = int(ratio * base.size)
        temp = np.sort(abs(base))
        idx = min(samples_to_clip + 1, base.size)
        clip_level = temp[0, -idx]

        if clip_level == 0:
            return base

        base = Clip(
            threshold=to_db(clip_level),
            soft=soft,
            normalize=normalize,
        )(base)

        return base


class Compose(Base):
    r"""Compose several transforms together.

    Args:
        transforms: list of transforms to compose
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([0.5, -0.5, 0.5, -0.5])
        >>> Compose([GainStage(12), Clip()])(base)
        array([ 1., -1.,  1., -1.], dtype=float32)

    """
    def __init__(
            self,
            transforms: Sequence[Base],
            *,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.transforms = transforms

    def _call(self, base: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            base = transform(base)
        return base


class CompressDynamicRange(Base):
    r"""Compress the dynamic range.

    The dynamic range of the signal
    is compressed
    by attenuating any sample
    whose amplitude exceeds a certain ``threshold_db``.
    The intensity of the attenuation
    is determined
    by the ``ratio`` parameter
    (the higher the ratio, the stronger the gain reduction).
    To avoid heavy distortion,
    the gain reduction is smoothed over time with a contour
    that is governed by the ``attack_time``
    and the ``release_time`` parameters.

    The input-output characteristic
    also features a non-linear region ("knee")
    around the threshold.
    The width of this region
    is controlled by the ``knee_radius_db`` parameter
    (expressed in decibels, and in absolute value):
    the nonlinear region is entered
    when the input signal exceeds a level
    given by ``threshold_db - kneeRadius_db``,
    hence some gain reduction
    can be also seen before hitting the main threshold,
    if the knee radius is greater than zero.

    Optionally,
    the resulting signal
    can be amplified (linearly)
    by means of the ``makeup_db`` gain parameter
    (expressed in decibels).
    Sample values can also be clipped
    to the interval ``[-1.0, 1.0]``
    when exceeding this range:
    this behaviour is achieved
    by setting the ``clip`` argument.

    .. note:: Setting ``makeup_db`` to ``None``
        triggers a special behaviour,
        for which the makeup gain is computed automatically
        in a way that the peak level of the processed signal
        is equal to the original peak level
        (before compression).

    Args:
        threshold_db: threshold in decibels
        ratio: ratio (the higher the ratio, the stronger the gain reduction)
        attack_time: attack time in seconds
        release_time: release time in seconds
        makeup_db: optional amplification gain
        clip: clip signal
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([0.5, 0.5], dtype='float32')
        >>> CompressDynamicRange(-6, 0.5)(base).round(4)
        array([0.5004, 0.5008], dtype=float32)

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
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_time = attack_time
        self.release_time = release_time
        self.knee_radius_db = knee_radius_db
        self.makeup_db = makeup_db
        self.clip = clip

    def _call(self, base: np.ndarray) -> np.ndarray:

        threshold_db = observe.observe(self.threshold_db)
        ratio = observe.observe(self.ratio)
        attack_time = observe.observe(self.attack_time)
        release_time = observe.observe(self.release_time)
        knee_radius_db = observe.observe(self.knee_radius_db)
        peak_db = to_db(get_peak(base))
        if self.makeup_db is None:
            makeup_db = None
            normalize_db = peak_db
        else:
            makeup_db = observe.observe(self.makeup_db)
            normalize_db = None
        clip = observe.observe(self.clip)

        # As values would be clipped by sox
        # when writing to the file,
        # we need to adjust the level
        # for clip=False
        if clip is False:
            if peak_db > 0:
                base = NormalizeByPeak(peak_db=0)(base)
                threshold_db = threshold_db - peak_db
                if normalize_db is None:
                    normalize_db = peak_db

        with tempfile.TemporaryDirectory() as tmp:
            infile = audeer.path(tmp, 'infile.wav')
            outfile = audeer.path(tmp, 'outfile.wav')
            cmd = [
                'sox',
                infile,
                outfile,
                'compand',
                f'{attack_time},{release_time}',
                f'{knee_radius_db}:{threshold_db},{ratio * threshold_db}',
            ]
            audiofile.write(infile, base, self.sampling_rate)
            audiofile.core.utils.run(cmd)
            base, _ = audiofile.read(
                outfile,
                always_2d=True,
                dtype=str(base.dtype),
            )

        if normalize_db is not None:
            base = NormalizeByPeak(peak_db=normalize_db)(base)
        if makeup_db is not None:
            base = from_db(makeup_db) * base

        if clip:
            base = Clip()(base)

        return base


class Fade(Base):
    r"""Fade-in and fade-out of signal.

    A fade is a gradual increase (fade-in)
    or decrease (fade-out)
    in the level
    of an audio signal.
    If ``in_db`` is greater than -120 dB
    the fade-in will start from the corresponding level,
    otherwise from silence.
    If ``out_db`` is greater than -120 dB
    the fade-out will end at the corresponding level,
    otherwise from silence.

    The shape of the fade-in and fade-out
    is selected via ``in_shape`` and ``out_shape``.
    The following figure shows all available shapes
    by the example of a fade-in.

    .. jupyter-execute::
        :hide-code:

        import auglib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import numpy as np
        import seaborn as sns

        for shape in auglib.core.transform.SUPPORTED_FADE_SHAPES:
            transform = auglib.transform.Fade(
                in_dur=101,
                in_shape=shape,
                out_dur=0,
                unit='samples',
            )
            augment = auglib.Augment(transform)
            augmented_signal = augment(np.ones(101), 1000)
            plt.plot(augmented_signal[0], label=shape)
        plt.ylabel('Magnitude')
        plt.xlabel('Fade-in Length')
        plt.legend()
        plt.grid(alpha=0.4)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.tick_params(axis=u'both', which=u'both',length=0)
        fig = plt.gcf()
        fig.set_size_inches(5.12, 3.84)
        plt.xlim([-1.2, 100.2])
        plt.ylim([-0.02, 1])
        sns.despine(left=True, bottom=True)

    Args:
        in_dur: duration of fade-in
        out_dur: duration of fade-out
        in_shape: shape of fade-in
        out_shape: shape of fade-out
        in_db: level in dB the fade-in should start at,
            -120 dB or less is equivalent
            to an amplitude of 0
        out_db: level in dB the fade-out should end at,
            -120 dB or less is equivalent
            to an amplitude of 0
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`).
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``in_shape`` or ``out_shape``
            contains a non-supported value
        ValueError: if ``in_db`` or ``out_db``
            are greater or equal to 0

    Examples:
        >>> base = np.array([1, 1, 1, 1, 1, 1])
        >>> Fade(in_dur=1, out_dur=1, unit='samples')(base)
        array([0., 1., 1., 1., 1., 0.], dtype=float32)

    """
    def __init__(
            self,
            *,
            in_dur: Union[int, float, observe.Base, Time] = 0.1,
            out_dur: Union[int, float, observe.Base, Time] = 0.1,
            in_shape: str = 'tukey',
            out_shape: str = 'tukey',
            in_db: float = -120,
            out_db: float = -120,
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            unit=unit,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        for shape in [in_shape, out_shape]:
            if shape not in SUPPORTED_FADE_SHAPES:
                raise ValueError(
                    f"Unknown fade shape '{shape}'. "
                    "Supported designs are: "
                    f"{', '.join(SUPPORTED_FADE_SHAPES)}."
                )
        for level in [in_db, out_db]:
            if level >= 0:
                raise ValueError(
                    'Fading level needs to be below 0 dB, '
                    f'not {level} dB.'
                )
        self.in_dur = in_dur
        self.out_dur = out_dur
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_db = in_db
        self.out_db = out_db

    def _call(self, base: np.ndarray) -> np.ndarray:

        in_shape = observe.observe(self.in_shape)
        out_shape = observe.observe(self.out_shape)
        in_db = observe.observe(self.in_db)
        out_db = observe.observe(self.out_db)
        in_samples = self.to_samples(
            observe.observe(self.in_dur),
            length=base.shape[1],
        )
        out_samples = self.to_samples(
            observe.observe(self.out_dur),
            length=base.shape[1],
        )

        fade_in = audmath.window(in_samples, shape=in_shape, half='left')
        fade_out = audmath.window(out_samples, shape=out_shape, half='right')

        # Adjust start level of fade-in
        # and/or end level of fade-out
        # if requested
        if in_db > -120:
            offset = from_db(in_db)
            fade_in = fade_in * (1 - offset) + offset
        if out_db > -120:
            offset = from_db(out_db)
            fade_out = fade_out * (1 - offset) + offset

        fade_in_win = np.concatenate(
            [
                fade_in[:base.shape[1]],
                np.ones(
                    max(0, base.shape[1] - in_samples),
                    dtype=DTYPE,
                ),
            ]
        ).astype(DTYPE)
        fade_out_win = np.concatenate(
            [
                np.ones(
                    max(0, base.shape[1] - out_samples),
                    dtype=DTYPE,
                ),
                fade_out[-base.shape[1]:],
            ]
        ).astype(DTYPE)
        base = fade_in_win * fade_out_win * base

        return base


class FFTConvolve(Base):
    r"""Convolve signal with another signal.

    The convolution is done by a FFT-based approach.

    Args:
        aux: auxiliary signal,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a signal with the same length
            as the base signal
            containing zeros
        keep_tail: keep the tail of the convolution result
            (extending the length of the signal),
            or to cut it out
            (keeping the original length of the input)
        transform: transformation applied to the auxiliary signal
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> aux = np.array([0, 1])
        >>> np.abs(FFTConvolve(aux)(base).round(4))
        array([0.,  1.,  2.,  3.,  4.], dtype=float32)
        >>> np.abs(FFTConvolve(aux, keep_tail=False)(base).round(4))
        array([0., 1., 2., 3.], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, np.ndarray, Base],
            *,
            keep_tail: Union[bool, observe.Base] = True,
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            aux=aux,
            transform=transform,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.keep_tail = keep_tail

    def _call(self, base: np.ndarray, aux: np.ndarray) -> np.ndarray:
        keep_tail = observe.observe(self.keep_tail)
        samples = base.shape[1]
        base = scipy.signal.fftconvolve(aux, base, mode='full')
        if not keep_tail:
            base = base[:, :samples]
        return base


class Function(Base):
    r"""Apply a custom function to the signal.

    The function gets as input a :class:`numpy.ndarray`
    of shape ``(channels, samples)``
    with the content of the audio signal
    and the sampling rate.
    Additional arguments can be provided with
    the ``function_args`` dictionary.
    Observable arguments
    (e.g. :class:`auglib.IntUni`)
    are automatically evaluated.
    The function must return a :class:`numpy.ndarray`.

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
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> def plus_c(x, sr, c):
        ...     return x + c
        >>> Function(plus_c, {'c': 1})(base)
        array([2., 3., 4., 5.], dtype=float32)
        >>> def halve(x, sr):
        ...     return x[:, ::2]
        >>> Function(halve)(base)
        array([1., 3.], dtype=float32)
        >>> Function(lambda x, sr: x * 2)(base)
        array([2., 4., 6., 8.], dtype=float32)

    """
    @audobject.init_decorator(
        resolvers={
            'function': audobject.resolver.Function,
        },
    )
    def __init__(
            self,
            function: Callable[..., np.ndarray],
            function_args: typing.Dict = None,
            *,
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.function = function
        self.function_args = function_args

    def _call(self, base: np.ndarray):

        # evaluate function arguments
        args = {}
        if self.function_args:
            for key, value in self.function_args.items():
                args[key] = observe.observe(value)

        base = self.function(base, self.sampling_rate, **args)

        return base


class GainStage(Base):
    r"""Scale signal by linear factor.

    If ``max_peak_db`` is not ``None``,
    the gain will be adjusted
    to never exceed the maximum.

    If ``clip`` is ``True``
    the resulting peak never exceeds 0 dB,
    even when ``max_peak_db`` > 0dB.

    Args:
        gain_db: amplification in decibels
        max_peak_db: maximum peak level allowed in decibels (see note)
        clip: clip sample values to the interval [-1.0, 1.0]
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> GainStage(to_db(2))(base)
        array([2., 4., 6., 8.], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.gain_db = gain_db
        self.max_peak_db = max_peak_db
        self.clip = clip

    def _call(self, base: np.ndarray) -> np.ndarray:
        gain_db = observe.observe(self.gain_db)
        clip = observe.observe(self.clip)
        max_peak_db = observe.observe(self.max_peak_db)

        if max_peak_db is not None:
            current_peak_db = to_db(get_peak(base))
            new_peak_db = gain_db + current_peak_db
            if new_peak_db > max_peak_db:
                gain_db = max_peak_db - current_peak_db
            base = GainStage(gain_db)(base)

        else:
            gain = from_db(gain_db)
            base = gain * base

        if clip:
            base = Clip(threshold=0)(base)

        return base


class HighPass(Base):
    r"""Run audio signal through a high-pass filter.

    Args:
        cutoff: cutoff frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Examples:
        >>> base = np.array([1, 2])
        >>> HighPass(7000, order=4)(base)
        array([ 0.0009335 , -0.00464588], dtype=float32)

    """
    def __init__(
            self,
            cutoff: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray) -> np.ndarray:
        cutoff = observe.observe(self.cutoff)
        order = observe.observe(self.order)
        if self.design == 'butter':
            b, a = scipy.signal.butter(
                order,
                cutoff,
                btype='highpass',
                fs=self.sampling_rate,
            )
            base = scipy.signal.lfilter(b, a, base)

        return base.astype(DTYPE)


class LowPass(Base):
    r"""Run audio signal through a low-pass filter.

    Args:
        cutoff: cutoff frequency in Hz
        order: filter order
        design: filter design,
            at the moment only ``'butter'`` is available
            corresponding to a Butterworth filter
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``design`` contains a non-supported value

    Examples:
        >>> base = np.array([1, 2])
        >>> LowPass(100)(base)
        array([0.01925927, 0.07629526], dtype=float32)

    """
    def __init__(
            self,
            cutoff: Union[float, observe.Base],
            *,
            order: Union[int, observe.Base] = 1,
            design: str = 'butter',
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray) -> np.ndarray:
        cutoff = observe.observe(self.cutoff)
        order = observe.observe(self.order)
        if self.design == 'butter':
            b, a = scipy.signal.butter(
                order,
                cutoff,
                btype='lowpass',
                fs=self.sampling_rate,
            )
            base = scipy.signal.lfilter(b, a, base)

        return base.astype(DTYPE)


class Mask(Base):
    r"""Masked transformation.

    Usually a transformation is applied on the whole signal.
    With :class:`auglib.transform.Mask` it is possible
    to mask the transformation within specific region(s).
    By default,
    regions outside the mask are augmented.
    If ``invert`` is set to ``True``,
    regions that fall inside the mask are augmented.

    Args:
        transform: transform object
        start_pos: start masking at this point (see ``unit``).
        duration: apply masking for this duration (see ``unit``).
            If set to ``None``
            masking is applied until the end of the signal
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
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 1, 1, 1, 1, 1])
        >>> Mask(
        ...     GainStage(to_db(2)),
        ...     start_pos=2,
        ...     duration=3,
        ...     unit='samples',
        ... )(base)
        array([2., 2., 1., 1., 1., 2.], dtype=float32)
        >>> Mask(
        ...     GainStage(to_db(2)),
        ...     start_pos=2,
        ...     duration=3,
        ...     unit='samples',
        ...     invert=True,
        ... )(base)
        array([1., 1., 2., 2., 2., 1.], dtype=float32)
        >>> Mask(
        ...     GainStage(to_db(2)),
        ...     step=2,
        ...     unit='samples',
        ... )(base)
        array([1., 1., 2., 2., 1., 1.], dtype=float32)
        >>> Mask(
        ...     GainStage(to_db(2)),
        ...     step=(2, 1),
        ...     unit='samples',
        ... )(base)
        array([1., 1., 2., 1., 1., 2.], dtype=float32)

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
            sampling_rate: int = 16000,
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
            sampling_rate=sampling_rate,
            unit=unit,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.transform = transform
        self.start_pos = start_pos
        self.duration = duration
        self.step = step
        self.invert = invert

    def _call(self, base: np.ndarray) -> np.ndarray:

        # store original signal, then apply transformation
        org_signal = base.copy()
        base = self.transform(base)
        num_samples = min(org_signal.shape[1], base.shape[1])

        # figure start and end position of mask
        start_pos = self.to_samples(
            self.start_pos,
            length=num_samples,
        )
        if self.duration is None:
            end_pos = num_samples
        else:
            end_pos = start_pos + self.to_samples(
                self.duration,
                length=num_samples,
            )
            end_pos = min(end_pos, num_samples)

        # create mask
        mask = np.zeros((base.shape[0], num_samples), dtype=bool)
        mask[:, :start_pos] = True
        mask[:, end_pos:] = True

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
                step = self.to_samples(
                    step,
                    length=num_samples,
                )
                step = min(step, end_pos - start_pos)
                # if we are not in a masked region, revert changes
                if not masked_region and step > 0:
                    mask[:, start_pos:start_pos + step] = True
                # increment position and switch condition
                start_pos += step
                masked_region = not masked_region

        # apply mask
        # invert = False -> remove augmentation within mask
        # invert = True -> keep augmentation within mask
        if not observe.observe(self.invert):
            mask = ~mask
        base[:, :num_samples][mask] = org_signal[:, :num_samples][mask]

        return base


class Mix(Base):
    r"""Mix two audio signals.

    Mix a base and auxiliary signal
    which may differ in length,
    but must have the same sampling rate.
    Individual gains can be set for both signals
    (``gain_base_db`` and ``gain_aux_db``).
    If ``snr_db`` is specified,
    ``gain_aux_db`` is automatically calculated
    to match the requested signal-to-noise ratio.
    The signal-to-noise ratio
    refers only to the overlapping parts
    of the base and auxiliary signal.

    ``write_pos_base`` specifies the starting point
    of adding the auxiliary signal
    to the the base signal.
    Selecting a sub-segment of the auxiliary signal is possible
    with selecting a starting point (``read_pos_aux``)
    and/or specifying its duration (``read_dur_aux``).

    In order to allow the looping of the auxiliary signal
    or its selected sub-segment,
    the ``loop_aux`` argument can be used.
    In case the auxiliary signal ends beyond
    the original ending point,
    the extra portion will be discarded,
    unless ``extend_base`` is set,
    in which case the base signal is extended accordingly.

    By default,
    the auxiliary signal is mixed
    into the base signal exactly once.
    However, the number of repetitions
    can be controlled with ``num_repeat``.
    Usually,
    this only makes sense
    when reading from random positions
    or random files.

    Args:
        aux: auxiliary signal,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a signal with the same length
            as the base signal
            containing zeros
        gain_base_db: gain of base signal
        gain_aux_db: gain of auxiliary signal.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise (base-to-aux) ratio in decibels
        write_pos_base: write position of base signal (see ``unit``)
        read_pos_aux: read position of auxiliary signal (see ``unit``)
        read_dur_aux: duration to read from auxiliary signal
            (see ``unit``).
            Set to ``None`` or ``0`` to read the whole signal
        clip_mix: clip amplitude values of mixed signal to [-1, 1]
        loop_aux: loop auxiliary signal if shorter than base signal
        extend_base: if needed, extend base signal to total required
            length (considering length of auxiliary signal)
        num_repeat: number of repetitions
        sampling_rate: sampling rate in Hz
        unit: literal specifying the format of ``write_pos_base``,
            ``read_pos_aux`` and ``read_dur_aux``
            (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary signal
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> aux = np.array([1, 1])
        >>> Mix(aux, num_repeat=2)(base)
        array([3., 4., 3., 4.], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, np.ndarray, Base],
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
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):

        super().__init__(
            aux=aux,
            num_repeat=num_repeat,
            sampling_rate=sampling_rate,
            unit=unit,
            transform=transform,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray, aux: np.ndarray) -> np.ndarray:
        write_pos_base = self.to_samples(
            self.write_pos_base,
            length=base.shape[1],
        )
        read_pos_aux = self.to_samples(
            self.read_pos_aux,
            length=aux.shape[1],
        )
        read_dur_aux = self.to_samples(
            self.read_dur_aux,
            length=aux.shape[1],
        )
        gain_base_db = observe.observe(self.gain_base_db)
        clip_mix = observe.observe(self.clip_mix)
        loop_aux = observe.observe(self.loop_aux)
        extend_base = observe.observe(self.extend_base)

        channels = 1  # only mono supported at the moment

        length_base = base.shape[1]
        length_aux = aux.shape[1]

        if clip_mix:
            clip = Clip()

        # Apply gain to base signal
        base = from_db(gain_base_db) * base
        if clip_mix:
            base = clip(base)

        # If read_dur_aux was `None` or `0`
        # we set read_dur_aux until the end of the signal
        orig_read_dur_aux = read_dur_aux
        if read_dur_aux == 0:
            read_dur_aux = length_aux - read_pos_aux

        # Find ending point of mixing for base
        end_pos_base = write_pos_base + read_dur_aux
        if end_pos_base > length_base:
            if extend_base:
                # Extend base signal by padding with zeros
                zeros = np.zeros(
                    (channels, end_pos_base - length_base),
                    dtype=DTYPE,
                )
                base = np.concatenate([base, zeros], axis=1)
            else:
                end_pos_base = length_base
        elif loop_aux:
            if orig_read_dur_aux == 0:
                end_pos_base = length_base  # allow looping

        # Adjust aux signal to match required length
        aux = aux[:, read_pos_aux:read_pos_aux + read_dur_aux]
        required_length_aux = end_pos_base - write_pos_base
        if aux.shape[1] < required_length_aux and loop_aux:
            repetitions = int(np.ceil(required_length_aux / aux.shape[1]))
            aux = np.tile(aux, repetitions)

        # Find end of mixing based on available aux signal
        if required_length_aux > aux.shape[1]:
            end_pos = aux.shape[1]
        else:
            end_pos = required_length_aux

        # Estimate gain for aux signal
        if self.snr_db is not None:

            # Estimate gain for aux signal
            # considering only overlapping parts of the signals
            snr_db = observe.observe(self.snr_db)

            # Ignore zero padded signal parts
            rms_start_base = write_pos_base
            rms_end_base = min([length_base, end_pos + write_pos_base])
            rms_start_aux = 0
            rms_end_aux = min([length_base - write_pos_base, end_pos])

            base_db = rms_db(base[:, rms_start_base:rms_end_base])
            aux_db = rms_db(aux[:, rms_start_aux:rms_end_aux])
            gain_aux_db = get_noise_gain_from_snr(base_db, aux_db, snr_db)
        else:
            gain_aux_db = observe.observe(self.gain_aux_db)

        base[:, write_pos_base:write_pos_base + end_pos] += (
            from_db(gain_aux_db) * aux[:, :end_pos]
        )
        if clip_mix:
            base = clip(base)

        return base


class NormalizeByPeak(Base):
    r"""Peak-normalize the signal to a desired level.

    Args:
        peak_db: desired peak value in decibels
        clip: clip sample values to the interval [-1.0, 1.0]
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([0.5, -0.5, 0.5, -0.5])
        >>> NormalizeByPeak()(base)
        array([ 1., -1.,  1., -1.], dtype=float32)
        >>> NormalizeByPeak(peak_db=-3)(base).round(4)
        array([ 0.7079, -0.7079,  0.7079, -0.7079], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.peak_db = peak_db
        self.clip = clip

    def _call(self, base: np.ndarray) -> np.ndarray:
        peak_db = observe.observe(self.peak_db)
        clip = observe.observe(self.clip)
        current_peak = get_peak(base)

        if current_peak == 0:
            return base

        gain_db = peak_db - to_db(current_peak)
        base = GainStage(gain_db, clip=clip)(base)

        return base


class PinkNoise(Base):
    r"""Adds pink noise.

    Args:
        gain_db: gain in decibels
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> seed(0)
        >>> base = np.array([0, 0, 0, 0])
        >>> PinkNoise()(base).round(4)
        array([ 0.4376, -1.    , -0.3993,  0.9617], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.gain_db = gain_db
        self.snr_db = snr_db

    def _call(self, base: np.ndarray) -> np.ndarray:

        noise = self._pink_noise(base.shape[1])
        noise = NormalizeByPeak()(noise)

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
            noise_db = rms_db(noise)
            signal_db = rms_db(base)
            gain_db = get_noise_gain_from_snr(signal_db, noise_db, snr_db)
        else:
            gain_db = observe.observe(self.gain_db)

        base = base + from_db(gain_db) * noise

        return base

    def _pink_noise(self, samples: int) -> np.ndarray:
        r"""Generate pink noise signal."""

        def psd(f):
            return 1 / np.where(f == 0, float('inf'), np.sqrt(f))

        white_noise = np.fft.rfft(np.random.randn(samples))

        # Normalized pink noise shape
        pink_shape = psd(np.fft.rfftfreq(samples))
        pink_shape = pink_shape / np.sqrt(np.mean(pink_shape ** 2))

        white_noise_shaped = white_noise * pink_shape
        pink_noise = np.fft.irfft(white_noise_shaped)

        return np.atleast_2d(pink_noise)


class Prepend(Base):
    r"""Prepend an auxiliary signal to the base signal.

    Args:
        aux: auxiliary signal,
            file,
            or signal generating transform.
            If a transform is given
            it will be applied
            to a signal with the same length
            as the base signal
            containing zeros
        read_pos_aux: read position of auxiliary signal
            (see ``unit``)
        read_dur_aux: duration to read from auxiliary signal
            (see ``unit``).
            Set to ``None`` or ``0`` to read the whole signal
        unit: literal specifying the format
            of ``read_pos_aux`` and ``read_dur_aux``
            (see :meth:`auglib.utils.to_samples`)
        transform: transformation applied to the auxiliary signal
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> aux = np.array([5, 6])
        >>> Prepend(aux)(base)
        array([5., 6., 1., 2., 3., 4.], dtype=float32)

    """
    def __init__(
            self,
            aux: Union[str, observe.Base, np.ndarray, Base],
            *,
            read_pos_aux: Union[int, float, observe.Base, Time] = 0.0,
            read_dur_aux: Union[int, float, observe.Base, Time] = None,
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            transform: Base = None,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            aux=aux,
            sampling_rate=sampling_rate,
            unit=unit,
            transform=transform,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.read_pos_aux = read_pos_aux
        self.read_dur_aux = read_dur_aux or 0

    def _call(self, base: np.ndarray, aux: np.ndarray) -> np.ndarray:
        if self.read_pos_aux != 0:
            read_pos_aux = self.to_samples(self.read_pos_aux)
        else:
            read_pos_aux = 0
        if self.read_dur_aux == 0:
            read_dur_aux = aux.shape[1] - read_pos_aux
        else:
            read_dur_aux = self.to_samples(self.read_dur_aux)
        base = np.concatenate(
            [aux[:, read_pos_aux:read_pos_aux + read_dur_aux], base],
            axis=1,
        )
        return base


class PrependValue(Base):
    r"""Prepend base signal with a constant value.

    Args:
        duration: duration of signal with constant value
            that will be prepended
            (see ``unit``)
        value: value to prepend
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1., 2., 3., 4.])
        >>> PrependValue(2, 5, unit='samples')(base)
        array([5., 5., 1., 2., 3., 4.], dtype=float32)

    """
    def __init__(
            self,
            duration: Union[int, float, observe.Base, Time],
            value: Union[float, observe.Base] = 0,
            *,
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            unit=unit,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.duration = duration
        self.value = value

    def _call(self, base: np.ndarray) -> np.ndarray:
        if self.duration != 0:
            samples = self.to_samples(self.duration)
            aux = np.ones(samples, dtype=DTYPE) * self.value
            base = Prepend(aux)(base)
        return base


class Resample(Base):
    r"""Resample signal to another sampling rate.

    Changes number of samples of the audio signal
    by applying :func:`audresample.resample` to the signal.

    Args:
        target_rate: target rate in Hz
        sampling_rate: current sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([0., 0., 0., 0.])
        >>> Resample(8000, sampling_rate=16000)(base)
        array([0., 0.], dtype=float32)

    """
    def __init__(
            self,
            target_rate: typing.Union[int, observe.List],
            *,
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: typing.Union[float, observe.Base] = None,
            **kwargs,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.target_rate = target_rate
        if 'override' in kwargs:
            self.override = kwargs['override']
            warnings.warn(
                "'override' argument is ignored "
                "and will be removed with version 1.2.0.",
                category=UserWarning,
                stacklevel=2,
            )

    def _call(self, base: np.ndarray) -> np.ndarray:

        original_rate = self.sampling_rate
        target_rate = observe.observe(self.target_rate)

        if original_rate != target_rate:
            base = audresample.resample(
                base,
                original_rate,
                target_rate,
            )

        return base


class Select(Base):
    r"""Randomly select from a pool of transforms.

    Args:
        transforms: list of transforms to choose from
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> seed(0)
        >>> base = np.array([1, 2, 3, 4])
        >>> aux = np.array([0., 0.])
        >>> Select([Prepend(aux), Append(aux)])(base)
        array([0., 0., 1., 2., 3., 4.], dtype=float32)

    """
    def __init__(
            self,
            transforms: Sequence[Base],
            *,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.transforms = transforms

    def _call(self, base: np.ndarray) -> np.ndarray:
        idx = np.random.randint(len(self.transforms))
        base = self.transforms[idx](base)
        return base


class Shift(Base):
    r"""Shift signal without changing its duration.

    The signal will be read
    from the position
    given by ``duration``,
    and the skipped samples from the beginng
    will be added at the end.

    Args:
        duration: duration of shift
        sampling_rate: sampling rate in Hz
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> Shift(1, unit='samples')(base)
        array([2., 3., 4., 1.], dtype=float32)

    """
    def __init__(
            self,
            duration: typing.Union[int, float, observe.Base, Time] = None,
            *,
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: typing.Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            unit=unit,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.duration = duration

    def _call(self, base: np.ndarray) -> np.ndarray:
        if self.duration is not None:
            duration = self.to_samples(
                observe.observe(self.duration),
                length=base.shape[1],
            )
            # Allow shift values that are out-of-bound
            # of the actual signal duration
            start_pos = duration % base.shape[1]
            if start_pos == 0:
                return base
            # Append data
            # that will be removed at the beginning
            base = np.concatenate(
                [base, base[:, :start_pos]],
                axis=1,
            )
            # Remove data from beginning
            base = base[:, start_pos:]

        return base


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
        sampling_rate: sampling rate in Hz
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Raises:
        ValueError: if ``shape`` contains a non-supported value

    Examples:
        >>> base = np.array([0, 0, 0, 0])
        >>> Tone(4000, shape='sine')(base).round(4)
        array([ 0.,  1., -0., -1.], dtype=float32)
        >>> Tone(4000, shape='square')(base)
        array([-1., -1.,  0.,  1.], dtype=float32)
        >>> Tone(4000, shape='sawtooth')(base)
        array([-1. , -0.5,  0. ,  0.5], dtype=float32)
        >>> Tone(4000, shape='triangle')(base)
        array([ 1.,  0., -1.,  0.], dtype=float32)

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
            sampling_rate: int = 16000,
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray) -> np.ndarray:
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
            tone_db = 20 * np.log10(rms)
            signal_db = rms_db(base)
            gain_db = get_noise_gain_from_snr(signal_db, tone_db, snr_db)
        else:
            gain_db = observe.observe(self.gain_db)

        if np.isnan(lfo_rate) or lfo_rate == 0:
            lfo_range = 0.0  # this will result in no modulation
            lfo_rate = 1.0  # dummy value to avoid NAN or division by zero

        period = self.sampling_rate / freq  # in samples
        lfo_period = self.sampling_rate / lfo_rate  # in samples
        omega = 2.0 * math.pi / period  # in radians per sample
        lfo_omega = 2.0 * math.pi / lfo_period  # in radians per sample
        lfo_amp = math.pi * lfo_range / self.sampling_rate  # half range (rad)

        time = np.array(range(base.shape[1]), dtype=DTYPE)
        gain = from_db(gain_db)

        if self.shape == 'sine':
            phase = 0.0
            phase = (
                omega * time
                + (lfo_amp * np.sin(lfo_omega * time) / lfo_omega)
            )
            base = base + gain * np.sin(phase)

        else:  # if shape is not sine, different approach

            current_period = period
            acc = 0.0

            # TODO: find solution without loop
            for sample in range(base.shape[1]):
                # Update the current period and compute the output sample
                current_period = (
                    2.0 * math.pi
                    / (omega + lfo_amp * np.sin(lfo_omega * sample))
                )
                increment = 1.0 / current_period
                if self.shape == 'square':
                    if acc > 0.5:
                        state = 1.0
                    elif acc < 0.5:
                        state = -1.0
                    else:
                        state = 0.0
                    base[:, sample] += gain * state
                elif self.shape == 'triangle':
                    base[:, sample] += 4 * gain * (np.abs(acc - 0.5) - 0.25)
                elif self.shape == 'sawtooth':
                    base[:, sample] += 2 * gain * (acc - 0.5)
                acc += increment  # 'phase-like' variable increasing in time
                if acc > 1.0:
                    acc = acc - np.floor(acc)  # phase wraps after each period

        return base


class Trim(Base):
    r"""Trim, zero pad, and/or repeat signal.

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
            relative to the start of the input signal
            (see ``unit``)
        end_pos: end point of the trimmed region,
            relative to the end of the input signal
            (see ``unit``).
            The end point is counted backwards
            from the end of the signal.
            If ``end_pos=512`` samples,
            it will remove the last 512 samples of the input
        duration: target duration of the resulting signal
            (see ``unit``).
            If set to ``None`` or ``0``,
            the selected section extends
            until the end
            or the beginning
            of the original signal
        fill: fill strategy
            if the end and/or start point
            of the trimmed region
            exceeds the signal.
            Three strategies are available:
            ``'none'`` the signal is not extended,
            ``'zeros'`` the signal is filled up with zeros,
            ``'loop'`` the trimmed signal is repeated
        fill_pos: position at which the selected fill strategy applies.
            ``'right'`` adds samples to the right,
            ``'left'`` adds samples to the left,
            or ``'both'`` adds samples on both sides,
            equally distributed starting at the right
        sampling_rate: sampling rate in Hz
        unit: literal specifying the format
            of ``start_pos``,
            ``end_pos``,
            and ``duration``
            (see :meth:`auglib.utils.to_samples`)
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: Probability to bypass the transformation

    Raises:
        ValueError: if ``fill`` contains a non-supported value
        ValueError: if ``fill_pos`` contains a non-supported value

    Examples:
        >>> base = np.array([1, 2, 3, 4])
        >>> Trim(start_pos=1, unit='samples')(base)
        array([2., 3., 4.], dtype=float32)
        >>> Trim(end_pos=1, unit='samples')(base)
        array([1., 2., 3.], dtype=float32)
        >>> Trim(start_pos=None, duration=2, unit='samples')(base)
        array([2., 3.], dtype=float32)
        >>> Trim(start_pos=2, duration=4, unit='samples', fill='loop')(base)
        array([3., 4., 3., 4.], dtype=float32)
        >>> Trim(
        ...     start_pos=1,
        ...     end_pos=1,
        ...     duration=4,
        ...     unit='samples',
        ...     fill='zeros',
        ... )(base)
        array([2., 3., 0., 0.], dtype=float32)
        >>> Trim(
        ...     start_pos=3,
        ...     duration=4,
        ...     unit='samples',
        ...     fill='zeros',
        ...     fill_pos='both',
        ... )(base)
        array([0., 4., 0., 0.], dtype=float32)

    """
    def __init__(
            self,
            *,
            start_pos: Union[int, float, observe.Base, Time] = 0,
            end_pos: Union[int, float, observe.Base, Time] = None,
            duration: Union[int, float, observe.Base, Time] = None,
            fill: str = 'none',
            fill_pos: str = 'right',
            sampling_rate: int = 16000,
            unit: str = 'seconds',
            preserve_level: Union[bool, observe.Base] = False,
            bypass_prob: Union[float, observe.Base] = None,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            unit=unit,
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
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

    def _call(self, base: np.ndarray) -> np.ndarray:

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
            return base

        # Convert start_pos, end_pos, duration to samples
        # and check for meaningful values
        length = base.shape[1]
        if self.start_pos is not None:
            start_pos = self.to_samples(
                observe.observe(self.start_pos),
                length=length,
                allow_negative=True,
            )
            if start_pos >= length:
                raise ValueError(f"'start_pos' must be <{length}.")
            if start_pos < 0:
                raise ValueError("'start_pos' must be >=0.")
        if self.end_pos is not None:
            end_pos = self.to_samples(
                observe.observe(self.end_pos),
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
                duration = self.to_samples(
                    observe.observe(self.duration),
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
                    f"and 'sampling_rate' = {self.sampling_rate} Hz "
                    "would lead to an empty signal "
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
            duration = base.shape[1] - start_pos - end_pos

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
            if end_pos == 0:
                base = base[:, start_pos:]
            else:
                base = base[:, start_pos:-end_pos]

            # Check the difference in samples
            # between current signal
            # and desired duration
            difference = base.shape[1] - duration

            # Expand signal if too short
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
                    # Expand signal by zeros
                    prepend_array = np.zeros(
                        (1, prepend_samples),
                        dtype=DTYPE,
                    )
                    append_array = np.zeros(
                        (1, append_samples),
                        dtype=DTYPE,
                    )
                elif self.fill == 'loop':
                    # Repeat signal in the expanded parts
                    repetitions = (
                        int(
                            max(prepend_samples, append_samples)
                            / base.shape[1]
                        )
                        + 1
                    )
                    repeated_array = np.tile(base, repetitions)
                    prepend_array = repeated_array[:, -prepend_samples:]
                    append_array = repeated_array[:, :append_samples]

                if prepend_samples > 0:
                    base = np.concatenate(
                        [prepend_array, base],
                        axis=1,
                    )

                if append_samples > 0:
                    base = np.concatenate(
                        [base, append_array],
                        axis=1,
                    )

            # Set start_pos to 0 for final trim with provided duration
            start_pos = 0

        base = base[:, start_pos:start_pos + duration]

        return base


class WhiteNoiseGaussian(Base):
    r"""Adds Gaussian white noise.

    Args:
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        stddev: standard deviation
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> seed(0)
        >>> base = np.array([0, 0])
        >>> WhiteNoiseGaussian()(base)
        array([ 0.03771907, -0.03963146], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.gain_db = gain_db
        self.snr_db = snr_db
        self.stddev = stddev

    def _call(self, base: np.ndarray) -> np.ndarray:
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
            noise_db = 10 * np.log10(self.stddev ** 2)
            signal_db = rms_db(base)
            gain_db = get_noise_gain_from_snr(signal_db, noise_db, snr_db)
        else:
            gain_db = observe.observe(self.gain_db)
        stddev = observe.observe(self.stddev)
        noise_generator = np.random.default_rng(seed=get_seed())

        base += (
            from_db(gain_db)
            * noise_generator.normal(0, stddev, base.shape)
        )

        return base



class WhiteNoiseUniform(Base):
    """Adds uniform white noise.

    Args:
        gain_db: gain in decibels.
            Ignored if ``snr_db`` is not ``None``
        snr_db: signal-to-noise ratio in decibels
        preserve_level: if ``True``
            the root mean square value
            of the augmented signal
            will be the same
            as before augmentation
        bypass_prob: probability to bypass the transformation

    Examples:
        >>> seed(0)
        >>> base = np.array([0, 0])
        >>> WhiteNoiseUniform()(base)
        array([ 0.27392337, -0.46042657], dtype=float32)

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
            preserve_level=preserve_level,
            bypass_prob=bypass_prob,
        )
        self.gain_db = gain_db
        self.snr_db = snr_db

    def _call(self, base: np.ndarray) -> np.ndarray:
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
            noise_db = -4.77125
            signal_db = rms_db(base)
            gain_db = get_noise_gain_from_snr(signal_db, noise_db, snr_db)
        else:
            gain_db = observe.observe(self.gain_db)

        noise_generator = np.random.default_rng(seed=get_seed())

        base += (
            from_db(gain_db)
            * noise_generator.uniform(-1, 1, base.shape)
        )

        return base
