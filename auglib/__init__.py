from auglib import observe
from auglib import transform
from auglib import utils
from auglib.core.cache import clear_default_cache_root
from auglib.core.cache import default_cache_root
from auglib.core.config import config
from auglib.core.interface import Augment
from auglib.core.seed import seed
from auglib.core.time import Time


__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
