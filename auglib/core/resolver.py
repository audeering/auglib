import typing

import numpy as np

import audobject


class ArrayResolver(audobject.resolver.Base):
    r"""Raise error if an array is encoded."""

    def decode(
        self,
        value: typing.Any,
    ) -> typing.Any:
        return value

    def encode(
        self,
        value: typing.Any,
    ) -> typing.Any:
        if isinstance(value, np.ndarray):
            raise ValueError(
                f"Cannot serialize an instance of "
                f"{type(value)}. "
                f"As a workaround, "
                f"save signal to disk "
                f"and pass filename."
            )
        return value

    def encode_type(self) -> type:
        return object


class ObservableListResolver(audobject.resolver.Base):
    r"""Raise error if list containing a signal is encoded."""

    def decode(
        self,
        value: typing.Any,
    ) -> typing.Any:
        return value

    def encode(
        self,
        value: typing.MutableSequence[typing.Any],
    ) -> typing.Any:
        for v in value:
            if isinstance(v, np.ndarray):
                raise ValueError(
                    f"Cannot serialize list "
                    f"if it contains an instance of "
                    f"{type(v)}. "
                    f"As a workaround, "
                    f"save signal to disk "
                    f"and add filename."
                )
        return value

    def encode_type(self) -> type:
        return object
