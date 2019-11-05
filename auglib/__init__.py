from .common import *
from .buffer import *
from .observe import *
from . import source
from . import transform
from . import utils
from . import interface

__version__ = 'unknown'

# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    pkg_resources = None
finally:
    del pkg_resources
