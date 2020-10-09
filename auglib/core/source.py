from typing import Union, Sequence

import numpy as np

from auglib.core.observe import (
    Float,
    Str,
)
from auglib.core.buffer import (
    AudioBuffer,
    Source,
    Transform,
)


class FromArray(Source):
    r"""Create buffer from an array.

    .. note:: Input array will be copied and flatten.

    Args:
        x: array with audio samples
        sampling_rate: sampling rate in Hz

    Example:
        >>> with FromArray([1] * 5, 8000)() as buf:
        ...     buf
        array([1., 1., 1., 1., 1.], dtype=float32)

    """
    def __init__(self, x: Union[np.ndarray, Sequence[float]],
                 sampling_rate: int):
        self.x = x if isinstance(x, np.ndarray) else np.array(x)
        self.sampling_rate = sampling_rate

    def call(self) -> AudioBuffer:
        return AudioBuffer.from_array(self.x, self.sampling_rate)


class Read(Source):
    r"""Create :class:`auglib.AudioBuffer` from an audio file.

    Uses soundfile for WAV, FLAC, and OGG files. All other audio files are
    first converted to WAV by sox or ffmpeg.

    .. note:: Audio will be converted to mono with sample type ``float32``.

    Args:
        path: path to audio file
        root: optional root directory
        duration: return only a specified duration in seconds
        offset: start reading at offset in seconds
        transform: optional transformation

    """
    def __init__(self, path: Union[str, Str], *,
                 root: str = None,
                 duration: Union[float, Float] = None,
                 offset: Union[float, Float] = 0,
                 transform: Transform = None):
        self.path = path
        self.root = root
        self.duration = duration
        self.offset = offset
        self.transform = transform

    def call(self) -> AudioBuffer:
        buf = AudioBuffer.read(self.path,
                               root=self.root,
                               duration=self.duration,
                               offset=self.offset)
        if self.transform:
            self.transform(buf)
        return buf
