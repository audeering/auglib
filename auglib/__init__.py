from auglib import observe
from auglib import sink
from auglib import source
from auglib import transform
from auglib import utils
from auglib.core.buffer import AudioBuffer
from auglib.core.buffer import Sink
from auglib.core.buffer import Source
from auglib.core.buffer_legacy import Transform
from auglib.core.cache import clear_default_cache_root
from auglib.core.cache import default_cache_root
from auglib.core.config import config
from auglib.core.exception import set_exception_handling
from auglib.core.interface import Augment
from auglib.core.interface import NumpyTransform
from auglib.core.observe_legacy import FloatList
from auglib.core.observe_legacy import FloatNorm
from auglib.core.observe_legacy import FloatUni
from auglib.core.observe_legacy import IntList
from auglib.core.observe_legacy import IntUni
from auglib.core.observe_legacy import StrList
from auglib.core.seed import seed
from auglib.core.time import Time


__all__ = []


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
