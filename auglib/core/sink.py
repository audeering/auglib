from typing import Union

from auglib.core.observe import Str
from auglib.core.buffer import (
    AudioBuffer,
    Sink,
)


class Write(Sink):
    r"""Write buffer to file.

    Args:
        path: file name of output audio file. The format (WAV, FLAC, OGG)
            will be inferred from the file name
        root: optional root directory
        precision: precision of writen file, can be `'16bit'`, `'24bit'`,
            `'32bit'`. Only available for WAV files
        normalize (bool, optional): normalize audio data before writing

    """
    def __init__(self, path: Union[str, Str], *, root: str = None,
                 precision: str = '16bit', normalize: bool = False):
        self.path = path
        self.root = root
        self.precision = precision
        self.normalize = normalize

    def call(self, buf: AudioBuffer):
        buf.write(self.path, root=self.root, precision=self.precision,
                  normalize=self.normalize)


class Play(Sink):
    r"""Play back buffer.

    """
    def __init__(self):
        pass

    def call(self, buf: AudioBuffer):
        buf.play()
