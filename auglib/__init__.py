from auglib import sink
from auglib import source
from auglib import transform
from auglib import utils
from auglib.core.buffer import (
    AudioBuffer,
    Sink,
    Source,
    Transform,
)
from auglib.core.exception import (
    ExceptionHandling,
    LibraryException,
    LibraryExceptionWarning,
    set_exception_handling,
)
from auglib.core.interface import (
    AudioModifier,
    NumpyTransform,
)
from auglib.core.observe import (
    FloatNorm,
    FloatUni,
    IntUni,
    StrList,
)

__all__ = [
    'sink',
    'source',
    'transform',
    'utils',
    'AudioBuffer',
    'AudioModifier',
    'ExceptionHandling',
    'FloatNorm',
    'FloatUni',
    'IntUni',
    'LibraryException',
    'LibraryExceptionWarning',
    'NumpyTransform',
    'set_exception_handling',
    'StrList',
    'Transform',
]


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    pkg_resources = None
finally:
    del pkg_resources
