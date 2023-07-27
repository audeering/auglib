from typing import Union

import audeer

from auglib.core import observe
from auglib.core.buffer import AudioBuffer
from auglib.core.buffer import Sink


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='auglib.AudioBuffer.write',
)
class Write(Sink):  # pragma: no cover
    r"""Write buffer to file.

    Args:
        path: file name of output audio file. The format (WAV, FLAC, OGG)
            will be inferred from the file name
        root: optional root directory
        precision: precision of written file, can be `'16bit'`, `'24bit'`,
            `'32bit'`. Only available for WAV files
        normalize (bool, optional): normalize audio data before writing

    """
    def __init__(self, path: Union[str, observe.Base], *, root: str = None,
                 precision: str = '16bit', normalize: bool = False):
        self.path = path
        self.root = root
        self.precision = precision
        self.normalize = normalize

    def call(self, buf: AudioBuffer):
        buf.write(self.path, root=self.root, precision=self.precision,
                  normalize=self.normalize)


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='auglib.AudioBuffer.play',
)
class Play(Sink):  # pragma: no cover
    r"""Play back buffer."""
    def __init__(self):
        pass

    def call(self, buf: AudioBuffer):
        buf.play()
