import typing

import audeer

from auglib.core import (
    buffer,
    observe,
)


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='audeer.safe_path',
)
def safe_path(
        path: typing.Union[str, observe.Base],
        *,
        root: str = None,
) -> str:  # pragma: no cover
    r"""Turns ``path`` into an absolute path.

    Args:
        path: file path
        root: optional root directory

    """
    return buffer.safe_path(path, root=root)
