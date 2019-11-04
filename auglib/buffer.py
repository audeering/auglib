import numpy as np
from typing import Union, Sequence
import audiofile as af
import sounddevice as sd

from .api import lib
from .common import Object
from .observe import observe, Number, Int, Float, Str
from .utils import to_samples, safe_path, mk_dirs


class AudioBuffer(object):
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
        >>> from auglib import AudioBuffer
        >>> with AudioBuffer(5, 8000, 1.0, unit='samples') as buf:
        >>>     buf
        [1. 1. 1. 1. 1.]

    """
    def __init__(self, duration: Union[int, float, Number],
                 sampling_rate: int,
                 value: Union[float, Float] = None, *, unit: str = 'seconds'):
        length = to_samples(duration, sampling_rate, unit=unit)
        sampling_rate = sampling_rate
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

    @staticmethod
    def from_array(x: Union[np.ndarray, Sequence[float]],
                   sampling_rate: int) -> 'AudioBuffer':
        r"""Create buffer from an array.

        .. note:: The input array will be flatten and converted to ``float32``.

        Args:
            x: array with audio samples
            sampling_rate: sampling rate in Hz

        Example:
            >>> from auglib import AudioBuffer
            >>> import numpy as np
            >>> with AudioBuffer.from_array([1] * 5, 8000) as buf:
            >>>     buf
            [1. 1. 1. 1. 1.]

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
        mk_dirs(path)
        af.write(path, self.data, self.sampling_rate, precision=precision,
                 normalize=normalize)

    def play(self, wait: bool = True):
        r"""Play back buffer.

        Args:
            wait: pause until file has been played back

        """
        sd.play(self.data, self.sampling_rate)
        if wait:
            sd.wait()


class Source(Object):
    r"""Base class for objects that create an
    :class:`auglib.buffer.AudioBuffer`.

    """
    def call(self) -> AudioBuffer:
        raise(NotImplementedError())

    def __call__(self) -> AudioBuffer:
        return self.call()


class Transform(Object):
    r"""Base class for objects applying some sort of transformation to an
    :class:`auglib.buffer.AudioBuffer`.

    Args:
        bypass_prob: probability to bypass the transformation

    """
    def __init__(self, bypass_prob: Union[float, Float] = None):
        self.bypass_prob = bypass_prob

    def call(self, buf: AudioBuffer):
        raise NotImplementedError()

    def __call__(self, buf: AudioBuffer) -> AudioBuffer:
        bypass_prob = observe(self.bypass_prob)
        if bypass_prob is None or np.random.random_sample() >= bypass_prob:
            self.call(buf)
        return buf


class Sink(Object):
    r"""Base class for objects that consume an
    :class:`auglib.buffer.AudioBuffer`.

    """
    def call(self, buf: AudioBuffer):
        raise(NotImplementedError())

    def __call__(self, buf: AudioBuffer):
        self.call(buf)
