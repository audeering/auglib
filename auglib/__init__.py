from .buffer import *
from .noise import *
from .tone import *

__version__ = 'unknown'

# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    pkg_resources = None
finally:
    del pkg_resources
