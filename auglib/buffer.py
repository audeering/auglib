import numpy as np
from typing import Union
from enum import IntEnum
import ctypes
from functools import wraps

from .api import lib
from .utils import dur_to_samples


class FilterDesign(IntEnum):
    r"""
    * `BUTTERWORTH`: Butterworth filter design
    """
    BUTTERWORTH = 0


def _check_data_decorator(func):
    # Preserve docstring, see:
    # https://docs.python.org/3.6/library/functools.html#functools.wraps
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        old_ptr = ctypes.addressof(lib.AudioBuffer_data(self.obj).contents)
        old_length = lib.AudioBuffer_size(self.obj)
        func(*args, **kwargs)
        new_ptr = ctypes.addressof(lib.AudioBuffer_data(self.obj).contents)
        new_length = lib.AudioBuffer_size(self.obj)
        if old_ptr != new_ptr or old_length != new_length:
            length = lib.AudioBuffer_size(self.obj)
            self.data = np.ctypeslib.as_array(lib.AudioBuffer_data(self.obj),
                                              shape=(length,))
    return inner


class AudioBuffer(object):
    r"""Holds a chunk of audio.

    By default an audio buffer is initialized with zeros. See
    :doc:`api-noise` and :doc:`api-tone` for other initialization
    methods. Use :meth:`auglib.AudioBuffer.FromArray` to create an audio
    buffer from a :class:`numpy.ndarray`.

    Note: always call ``free()`` when a buffer is no longer needed. This
    will free the memory. Consider to use a ``with`` statement if possible.

    * attr:`obj` holds the underlying c object
    * attr:`sampling_rate` holds the sampling rate in Hz
    * attr:`data` holds the audio data as a :class:`numpy.ndarray`

    Args:
        duration: buffer duration (see ``unit``)
        sampling_rate: sampling rate in Hz
        unit: literal specifying the format of ``duration``
             (see :meth:`auglib.utils.dur2samples`)

    Example:
        >>> from auglib import AudioBuffer
        >>> with AudioBuffer(1.0, 8000) as buf:
        >>>     buf.data += 1
        >>>     buf
        [1. 1. 1. ... 1. 1. 1.]

    """
    def __init__(self, duration: float, sampling_rate: int,
                 *, unit: str = 'seconds'):
        length = dur_to_samples(duration, sampling_rate, unit=unit)
        self.obj = lib.AudioBuffer_new(length, sampling_rate)
        self.sampling_rate = sampling_rate
        self.data = np.ctypeslib.as_array(lib.AudioBuffer_data(self.obj),
                                          shape=(length, ))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def free(self):
        r"""Free the audio buffer.

        Note: Always call ``free()`` when an object is no longer needed to
        release its memory.

        """
        if self.obj:
            lib.AudioBuffer_free(self.obj)
            self.data = None
            self.obj = None

    @staticmethod
    def FromArray(x: np.ndarray, sampling_rate: int) -> 'AudioBuffer':
        r"""Create buffer from Numpy array.

        Note: The input array will be flatten.

        Args:
            x: a Numpy  :class:`numpy.ndarray`
            sampling_rate: sampling rate in Hz

        Example:
            >>> from auglib import AudioBuffer
            >>> import numpy as np
            >>> with AudioBuffer.FromArray(np.ones(5), 8000) as buf:
            >>>     buf
            [1. 1. 1. 1. 1.]
        """
        buf = AudioBuffer(x.size, sampling_rate, unit='samples')
        np.copyto(buf.data, x.flatten())  # takes care of data type
        return buf

    @_check_data_decorator
    def mix(self, aux_buf: 'AudioBuffer',
            *,
            gain_base_db: float = 0.0,
            gain_aux_db: float = 0.0,
            write_pos_base: Union[int, float, str] = 0,
            read_pos_aux: Union[int, float, str] = 0,
            read_dur_aux: Union[int, float, str] = 0,
            clip_mix: bool = False,
            loop_aux: bool = False,
            extend_base: bool = False,
            unit='seconds'):
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
        ``read_dur_aux`` (the total length of the selected segment). Note:
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
            aux_buf: auxiliary buffer
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
                (see :meth:`auglib.utils.dur2samples`)

        Example:
            >>> with AudioBuffer(1.0, 8000) as base
            >>>     with AudioBuffer(1.0, 8000) as aux:
            >>>         aux.data += 1
            >>>         base.mix(aux)
            >>>         base
            [1. 1. 1. ... 1. 1. 1.]

        """
        write_pos_base = dur_to_samples(write_pos_base, self.sampling_rate,
                                        unit=unit)
        read_pos_aux = dur_to_samples(read_pos_aux, aux_buf.sampling_rate,
                                      unit=unit)
        read_dur_aux = dur_to_samples(read_dur_aux, aux_buf.sampling_rate,
                                      unit=unit)
        lib.AudioBuffer_mix(self.obj, aux_buf.obj, gain_base_db, gain_aux_db,
                            write_pos_base, read_pos_aux, read_dur_aux,
                            clip_mix, loop_aux, extend_base)

    @_check_data_decorator
    def append(self, aux_buf: 'AudioBuffer', *, read_pos_aux: int = 0,
               read_dur_aux: int = 0, unit: str = 'seconds'):
        r"""Append an auxiliary buffer at the end of the base buffer.

        Base and auxiliary buffer may differ in length but must have the
        same sampling rate.

        Options are provided for selecting a specific portion of the
        auxiliary buffer (see ``readPos_aux`` and ``read_dur_aux``).
        After the operation is complete, the final length of the base buffer
        will be ``read_dur_aux`` samples greater then the original length.

        Args:
            aux_buf: auxiliary buffer
            read_pos_aux: read position of auxiliary buffer (see ``unit``)
            read_dur_aux: duration to read from auxiliary buffer (see
                ``unit``). Set to 0 to read the whole buffer.
            unit: literal specifying the format of ``read_pos_aux`` and
                ``read_dur_aux`` (see :meth:`auglib.utils.dur2samples`)

        >>> with AudioBuffer(1.0, 8000) as base:
        >>>     with AudioBuffer(1.0, 8000) as aux:
        >>>         aux.data += 1
        >>>         base.append(aux)
        >>>         base
        [0. 0. 0. ... 1. 1. 1.]

        """
        lib.AudioBuffer_append(self.obj, aux_buf.obj, read_pos_aux,
                               read_dur_aux)

    @_check_data_decorator
    def fft_convolve(self, aux_buf: 'AudioBuffer', *, keep_tail: bool = True):
        r"""Convolve the audio buffer (base) with another buffer (auxiliary)
        using an impulse response (FFT-based approach).

        Args:
            aux_buf: auxiliary buffer
            keep_tail: keep the tail of the convolution result (extending
            the length of the buffer), or to cut it out (keeping the
            original length of the input)

        """
        old_addr = ctypes.addressof(lib.AudioBuffer_data(self.obj).contents)
        old_length = lib.AudioBuffer_size(self.obj)
        lib.AudioBuffer_fftConvolve(self.obj, aux_buf.obj,
                                    keep_tail)
        new_addr = ctypes.addressof(lib.AudioBuffer_data(self.obj).contents)
        new_length = lib.AudioBuffer_size(self.obj)
        if old_addr != new_addr or old_length != new_length:
            length = lib.AudioBuffer_size(self.obj)
            self.data = np.ctypeslib.as_array(lib.AudioBuffer_data(self.obj),
                                              shape=(length,))

    def low_pass(self, order: int, cutoff: float,
                 design=FilterDesign.BUTTERWORTH):
        r"""Run audio buffer through a low-pass filter.

        Args:
            order: filter order
            cutoff: cutoff frequency in Hz
            design: filter design (see :class:`FilterType`)

        """
        if design == FilterDesign.BUTTERWORTH:
            lib.AudioBuffer_butterworthLowPassFilter(self.obj, order, cutoff)

    # TODO: bring this back to life once the highpass filter is fixed in auglib
    # def high_pass(self, order: int, cutoff: float,
    #               design=FilterDesign.BUTTERWORTH):
    #     r"""Run audio buffer through a high-pass filter.
    #
    #     Args:
    #         order: filter order
    #         cutoff: cutoff frequency in Hz
    #         design: filter design (see :class:`FilterType`)
    #
    #     """
    #     if design == FilterDesign.BUTTERWORTH:
    #         lib.AudioBuffer_butterworthHighPassFilter(self.obj, order, cutoff) # NOQA

    def band_pass(self, order: int, center: float, bandwidth: float,
                  design=FilterDesign.BUTTERWORTH):
        r"""Run audio buffer through a band-pass filter.

        Args:
            order: filter order
            center: center frequency in Hz
            bandwidth: bandwidth frequency in Hz
            design: filter design (see :class:`FilterType`)

        """
        if design == FilterDesign.BUTTERWORTH:
            lib.AudioBuffer_butterworthBandPassFilter(self.obj, order, center,
                                                      bandwidth)

    def clip(self, *, threshold: float = 0.0, soft: bool = False,
             as_ratio: bool = False, normalize: bool = False):
        r"""Hard/soft-clip the audio buffer.

        By default, ``threshold`` sets the amplitude level to which the signal
        is clipped. If this value is negative, it will be interpreted as
        expressed in decibels, otherwise it is interpreted as a sample
        amplitude. Unless ``as_ratio`` is set, in this case ``threshold`` is
        interpreted as a ratio between the number of samples that are to be
        clipped and the total number of samples in the buffer. The optional
        argument ``soft`` triggers a soft-clipping behaviour, for which the
        whole waveform is warped through a cubic non-linearity, resulting in
        a smooth transition between the flat (clipped) regions and the
        rest of the waveform.

        Args:
            threshold: clip threshold or clip ratio (see description)
            soft: apply soft-clipping (see description)
            as_ratio: clip by ratio (see description)
            normalize: after clipping normalize buffer to 0 decibels

        """
        lib.AudioBuffer_clip(self.obj, threshold, soft, as_ratio, normalize)

    def normalize_by_peak(self, *, peak_db: float = 0.0, clip: bool = False):
        r"""Peak-normalize the audio buffer to a desired level.

        Args:
            peak_db: desired peak value in decibels
            clip: clip sample values to the interval [-1.0, 1.0]

        """
        lib.AudioBuffer_normalizeByPeak(self.obj, peak_db, clip)

    def gain_stage(self, gain_db: float, *, clip: bool = False):
        r"""Scale the buffer by the linear factor that corresponds
        to ``gain_dB`` (in decibels).

        Args:
            gain_db: amplification in decibels
            clip: clip sample values to the interval [-1.0, 1.0]

        """
        lib.AudioBuffer_gainStage(self.obj, gain_db, clip)

    def __len__(self):
        return lib.AudioBuffer_size(self.obj)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __eq__(self, other: 'AudioBuffer'):
        return self.sampling_rate == other.sampling_rate and \
            np.array_equal(self.data, other.data)
