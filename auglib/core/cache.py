import os
import shutil

import audeer

from auglib.core.config import config
from auglib.core.interface import Augment


def clear_default_cache_root(augment: Augment = None):
    r"""Clear default cache directory.

    If ``augment`` is not None,
    deletes only the sub-directory where files
    created by the :class:`auglib.Augment` object are stored.

    Args:
        augment: optional augmentation object

    """
    root = default_cache_root(augment)
    if os.path.exists(root):
        shutil.rmtree(root)
    if augment is None:
        audeer.mkdir(root)


def default_cache_root(augment: Augment = None) -> str:
    r"""Path to default cache directory.

    The default cache directory defines
    the path where augmented files will be stored.
    It is given by the path specified
    by the environment variable
    ``AUGLIB_CACHE_ROOT``
    or by
    ``auglib.config.CACHE_ROOT``.

    If ``augment`` is not None,
    returns the sub-directory where files
    created by the :class:`auglib.Augment` object are stored.

    Args:
        augment: optional augmentation object

    Returns:
        cache directory path

    """
    root = os.environ.get("AUGLIB_CACHE_ROOT") or config.CACHE_ROOT
    if augment is not None:
        root = os.path.join(root, augment.short_id)
    return audeer.path(root, follow_symlink=True)
