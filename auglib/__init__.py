from auglib import sink
from auglib import source
from auglib import transform
from auglib import utils
from auglib import observe
from auglib.core.buffer import (
    AudioBuffer,
    Sink,
    Source,
    Transform,
)
from auglib.core.config import config
from auglib.core.exception import (
    ExceptionHandling,
    LibraryException,
    LibraryExceptionWarning,
    set_exception_handling,
)
from auglib.core.interface import (
    Augment,
    default_cache_root,
    clear_default_cache_root,
    NumpyTransform,
)
from auglib.core.observe_legacy import (
    FloatList,
    FloatNorm,
    FloatUni,
    IntList,
    IntUni,
    StrList,
)

__all__ = []


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
