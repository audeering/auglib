import os
from typing import Union, Sequence

import audiofile as af
import numpy as np

import audeer
import audobject

from auglib.core.api import lib
from auglib.core.observe import (
    Float,
    Number,
    observe,
    Str,
)
from auglib.core.utils import (
    assert_non_negative_number,
    to_samples,
    safe_path,
)
from auglib.core.exception import _check_exception_decorator


class AudioBuffer:
    r"""Holds a chunk of audio.

    By default an audio buffer is initialized with zeros. See ``value``
    argument and :doc:`api-source` for other ways.

    .. note:: Always call ``free()`` when a buffer is no longer needed. This
        will free the memory. Consider to use a ``with`` statement if possible.

    * :attr:`obj` holds the underlying C object
    * :attr:`sampling_rate` holds the sampling rate in Hz
    * :attr:`data` holds the audio data as a :class:`numpy.ndarray`

    Args:
        duration: buffer duration (see ``unit``)
        sampling_rate: sampling rate in Hz
        value: initialize buffer with a value or from a `numpy.ndarray` or
            (otherwise initialized with zeros)
        unit: literal specifying the format of ``duration``
            (see :meth:`auglib.utils.to_samples`)

    Example:

        >>> with AudioBuffer(5, 8000, 1.0, unit='samples') as buf:
        ...     buf
        array([1., 1., 1., 1., 1.], dtype=float32)

    """
    def __init__(self, duration: Union[int, float, Number],
                 sampling_rate: int,
                 value: Union[float, Float] = None, *, unit: str = 'seconds'):
        length = to_samples(duration, sampling_rate, unit=unit)
        assert_non_negative_number(sampling_rate)
        self.obj = lib.AudioBuffer_new(length, sampling_rate)
        self.sampling_rate = sampling_rate
        self.data = np.ctypeslib.as_array(lib.AudioBuffer_data(self.obj),
                                          shape=(length, ))
        if value:
            self.data += observe(value)

    def __len__(self):
        return lib.AudioBuffer_size(self.obj)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __eq__(self, other: 'AudioBuffer'):
        return self.sampling_rate == other.sampling_rate and \
            np.array_equal(self.data, other.data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def free(self):
        r"""Free the audio buffer.

        .. note:: Always call ``free()`` when an object is no longer needed to
            release its memory.

        """
        if self.obj:
            lib.AudioBuffer_free(self.obj)
            self.data = None
            self.obj = None

    def dump(self):
        r"""Dump audio buffer to stdout."""
        lib.AudioBuffer_dump(self.obj)

    @property
    def peak(self) -> float:
        r"""Return buffer peak."""
        if self.obj:
            return lib.AudioBuffer_getPeak(self.obj)

    @property
    def peak_db(self) -> float:
        r"""Return buffer peak in decibels."""
        if self.obj:
            return lib.AudioBuffer_getPeakDecibels(self.obj)

    @staticmethod
    def from_array(x: Union[np.ndarray, Sequence[float]],
                   sampling_rate: int) -> 'AudioBuffer':
        r"""Create buffer from an array.

        .. note:: The input array will be flatten and converted to ``float32``.

        Args:
            x: array with audio samples
            sampling_rate: sampling rate in Hz

        Example:
            >>> with AudioBuffer.from_array([1] * 5, 8000) as buf:
            ...     buf
            array([1., 1., 1., 1., 1.], dtype=float32)

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        buf = AudioBuffer(x.size, sampling_rate, unit='samples')
        np.copyto(buf.data, x.flatten())  # takes care of data type
        return buf

    @staticmethod
    def read(path: Union[str, Str], *, root: str = None,
             duration: Union[float, Float] = None,
             offset: Union[float, Float] = 0) -> 'AudioBuffer':
        r"""Create buffer from an audio file.

        Uses soundfile for WAV, FLAC, and OGG files. All other audio files are
        first converted to WAV by sox or ffmpeg.

        .. note:: Audio will be converted to mono with sample type ``float32``.

        Args:
            path: path to file
            root: optional root directory
            duration: return only a specified duration in seconds
            offset: start reading at offset in seconds.

        """
        path = safe_path(path, root=root)
        duration = observe(duration)
        offset = observe(offset)
        x, sr = af.read(path, duration=duration, offset=offset, always_2d=True)
        return AudioBuffer.from_array(x[0, :], sr)

    def write(self, path: Union[str, Str], *, root: str = None,
              precision: str = '16bit', normalize: bool = False):
        r"""Write buffer to a audio file.

        Args:
            path: file name of output audio file. The format (WAV, FLAC, OGG)
                will be inferred from the file name
            root: optional root directory
            precision: precision of writen file, can be `'16bit'`, `'24bit'`,
                `'32bit'`. Only available for WAV files
            normalize (bool, optional): normalize audio data before writing

        """
        path = safe_path(path, root=root)
        audeer.mkdir(os.path.dirname(path))
        af.write(path, self.data, self.sampling_rate, precision=precision,
                 normalize=normalize)

    def to_array(self) -> np.ndarray:
        r"""Returns copy of buffer data.

        Returns:
            buffer data

        """
        return self.data.copy()

    def play(self, wait: bool = True):
        r"""Play back buffer.

        Args:
            wait: pause until file has been played back

        """
        import sounddevice as sd
        sd.play(self.data, self.sampling_rate)
        if wait:
            sd.wait()


class Source(audobject.Object):
    r"""Base class for objects that create an
    :class:`auglib.AudioBuffer`.

    """
    def call(self) -> AudioBuffer:
        r"""Creates an :class:`auglib.AudioBuffer`.

        Raises:
            NotImplementedError: raised if not overwritten in child class

        """
        raise NotImplementedError()

    def __call__(self) -> AudioBuffer:
        return self.call()


class Transform(audobject.Object):
    r"""Base class for objects applying some sort of transformation to an
    :class:`auglib.AudioBuffer`.

    Args:
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, bypass_prob: Union[float, Float] = None):
        self.bypass_prob = bypass_prob

    def _call(self, buf: AudioBuffer):
        r"""Transforms an :class:`auglib.AudioBuffer`.

        Args:
            buf: audio buffer

        Raises:
            NotImplementedError: raised if not overwritten in child class

        """
        raise NotImplementedError()

    @_check_exception_decorator
    def __call__(self, buf: AudioBuffer) -> AudioBuffer:
        bypass_prob = observe(self.bypass_prob)
        if bypass_prob is None or np.random.random_sample() >= bypass_prob:
            self._call(buf)
        return buf


class Sink(audobject.Object):
    r"""Base class for objects that consume an
    :class:`auglib.AudioBuffer`.

    """
    def call(self, buf: AudioBuffer):
        r"""Consume an :class:`auglib.AudioBuffer`.

        Args:
            buf: audio buffer

        Raises:
            NotImplementedError: raised if not overwritten in child class

        """
        raise NotImplementedError()

    def __call__(self, buf: AudioBuffer):
        self.call(buf)
