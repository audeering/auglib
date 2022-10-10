import typing

import audobject

from auglib.core.buffer import AudioBuffer


class AudioBufferResolver(audobject.resolver.Base):
    r"""Raise error if a buffer is encoded."""

    def decode(
            self,
            value: typing.Any,
    ) -> typing.Any:
        return value

    def encode(
            self,
            value: typing.Any,
    ) -> typing.Any:
        if isinstance(value, AudioBuffer):
            raise ValueError(
                f"Cannot serialize an instance of "
                f"{type(value)}. "
                f"As a workaround, "
                f"save buffer to disk "
                f"and pass filename."
            )
        return value

    def encode_type(self) -> type:
        return object


class ObservableListResolver(audobject.resolver.Base):
    r"""Raise error if list containing a buffer is encoded."""

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
            if isinstance(v, AudioBuffer):
                raise ValueError(
                    f"Cannot serialize list "
                    f"if it contains an instance of "
                    f"{type(v)}. "
                    f"As a workaround, "
                    f"save buffer to disk "
                    f"and add filename."
                )
        return value

    def encode_type(self) -> type:
        return object
