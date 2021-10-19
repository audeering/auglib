import os
from typing import Union, Sequence
import warnings

import audiofile as af
import numpy as np

import audeer
import audobject

from auglib.core import observe
from auglib.core.api import lib
from auglib.core.utils import (
    assert_non_negative_number,
    to_samples,
)


class AudioBuffer:
    r"""Holds a chunk of audio.

    By default an audio buffer is initialized with zeros.
    See ``value`` argument for other ways.

    .. note:: Always call ``free()`` when a buffer is no longer needed.
        This will free the memory.
        Consider to use a ``with`` statement if possible.

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
        array([[1., 1., 1., 1., 1.]], dtype=float32)
        >>> with AudioBuffer.from_array([1] * 5, 8000) as buf:
        ...     buf
        array([[1., 1., 1., 1., 1.]], dtype=float32)

    """
    def __init__(
            self,
            duration: Union[int, float, observe.Base],
            sampling_rate: int,
            value: Union[float, observe.Base] = None,
            *,
            unit: str = 'seconds',
    ):
        assert_non_negative_number(sampling_rate)

        length = to_samples(duration, sampling_rate=sampling_rate, unit=unit)
        self._obj = lib.AudioBuffer_new(length, sampling_rate)
        self._data = np.ctypeslib.as_array(
            lib.AudioBuffer_data(self._obj),
            shape=(length, ),
        )

        self.sampling_rate = sampling_rate
        r"""Sampling rate in Hz."""

        if value:
            self._data += observe.observe(value)

    def __enter__(self):
        return self

    def __eq__(self, other: 'AudioBuffer'):
        return self.sampling_rate == other.sampling_rate and \
            np.array_equal(self._data, other._data)

    def __exit__(self, *args):
        self.free()

    def __len__(self):
        return lib.AudioBuffer_size(self._obj)

    def __repr__(self):
        return repr(self.to_array(copy=False))

    def __str__(self):
        return str(self.to_array(copy=False))

    @property
    def peak(self) -> float:
        r"""Buffer peak."""
        if self._obj:
            return lib.AudioBuffer_getPeak(self._obj)

    @property
    def peak_db(self) -> float:
        r"""Buffer peak in decibels."""
        if self._obj:
            return lib.AudioBuffer_getPeakDecibels(self._obj)

    def dump(self):
        r"""Dump audio buffer to stdout."""
        lib.AudioBuffer_dump(self._obj)

    def free(self):
        r"""Free the audio buffer.

        .. note:: Always call ``free()`` when an object is no longer needed to
            release its memory.

        """
        if self._obj:
            lib.AudioBuffer_free(self._obj)
            self._data = None
            self._obj = None

    def play(self, wait: bool = True):
        r"""Play back buffer.

        Args:
            wait: pause until file has been played back

        """
        import sounddevice as sd
        sd.play(self._data, self.sampling_rate)
        if wait:
            sd.wait()

    def to_array(
            self,
            *,
            copy: bool = True,
    ) -> np.ndarray:
        r"""Return buffer as :class:`numpy.ndarray`.

        By default,
        returns a copy of the buffer
        amd changes to the array
        will not affect the buffer.
        However,
        if ``copy`` is set to ``False``
        changes to the array
        will also alter the buffer.

        Args:
            copy: controls if a copy of the data should be returned

        Returns:
            array with shape ``(1, samples)``

        Example:
            >>> buf = AudioBuffer(5, 8000, unit='samples')
            >>> x = buf.to_array()
            >>> x
            array([[0., 0., 0., 0., 0.]], dtype=float32)
            >>> x.fill(1)
            >>> x.max()
            1.0
            >>> buf.peak
            0.0
            >>> y = buf.to_array(copy=False)
            >>> y
            array([[0., 0., 0., 0., 0.]], dtype=float32)
            >>> y.fill(1)
            >>> y.max()
            1.0
            >>> buf.peak
            1.0
            >>> buf.free()

        """  # noqa
        if copy:
            return self._data.copy().reshape(1, -1)
        else:
            return self._data.reshape(1, -1)

    @audeer.deprecated_keyword_argument(
        deprecated_argument='precision',
        removal_version='0.10.0',
    )
    def write(self, path: Union[str, observe.Base], *, root: str = None,
              bit_depth: int = 16, normalize: bool = False):
        r"""Write buffer to a audio file.

        Args:
            path: file name of output audio file. The format (WAV, FLAC, OGG)
                will be inferred from the file name
            root: optional root directory
            bit_depth: bit depth of written file in bit, can be 8, 16,
                24 for WAV and FLAC files, and in addition 32 for WAV files
            normalize (bool, optional): normalize audio data before writing

        """
        path = safe_path(path, root=root)
        audeer.mkdir(os.path.dirname(path))
        af.write(path, self._data, self.sampling_rate, bit_depth=bit_depth,
                 normalize=normalize)

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
            array([[1., 1., 1., 1., 1.]], dtype=float32)

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        buf = AudioBuffer(x.size, sampling_rate, unit='samples')
        np.copyto(buf._data, x.flatten())  # takes care of data type
        return buf

    @staticmethod
    def read(path: Union[str, observe.Base], *, root: str = None,
             duration: Union[float, observe.Base] = None,
             offset: Union[float, observe.Base] = 0) -> 'AudioBuffer':
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
        duration = observe.observe(duration)
        offset = observe.observe(offset)
        x, sr = af.read(path, duration=duration, offset=offset, always_2d=True)
        return AudioBuffer.from_array(x[0, :], sr)


# @audeer.deprecated(
#     removal_version='1.0.0',
#     alternative='auglib.AudioBuffer',
# )
# ->
# TypeError: function() argument 1 must be code, not str
# ->
# as a workaround we raise the deprecation warning in __init__
# This happens when deriving from this class.
class Source(audobject.Object):
    r"""Base class for objects that create an
    :class:`auglib.AudioBuffer`.

    """
    def __init__(self):
        message = (
            'Source is deprecated and will be removed '
            'with version 1.0.0. Use auglib.AudioBuffer instead.'
        )
        warnings.warn(message, category=UserWarning, stacklevel=2)

    def call(self) -> AudioBuffer:
        r"""Creates an :class:`auglib.AudioBuffer`.

        Raises:
            NotImplementedError: raised if not overwritten in child class

        """
        raise NotImplementedError()

    def __call__(self) -> AudioBuffer:
        return self.call()


# @audeer.deprecated(
#     removal_version='1.0.0',
#     alternative='auglib.AudioBuffer',
# )
# ->
# TypeError: function() argument 1 must be code, not str
# ->
# as a workaround we raise the deprecation warning in __init__
# This happens when deriving from this class.
class Sink(audobject.Object):
    r"""Base class for objects that consume an
    :class:`auglib.AudioBuffer`.

    """
    def __init__(self):
        message = (
            'Sink is deprecated and will be removed '
            'with version 1.0.0. Use auglib.AudioBuffer instead.'
        )
        warnings.warn(message, category=UserWarning, stacklevel=2)

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


def safe_path(path: Union[str, observe.Base], *, root: str = None) -> str:
    r"""Turns ``path`` into an absolute path.

    Args:
        path: file path
        root: optional root directory

    """
    path = observe.observe(path)
    if root:
        path = os.path.join(root, path)
    return audeer.safe_path(path)
