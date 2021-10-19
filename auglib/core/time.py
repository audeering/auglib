import typing

import humanfriendly as hf

import audobject

from auglib.core.seed import seed
from auglib.core import observe


class Time(audobject.Object):
    r"""Represent timestamp or timespan.

    Different time formats are supported,
    but calling the object always returns
    the value expressed as number of samples.

    Args:
        value: timestamp or timespan
        unit: literal specifying the format
            (see :meth:`auglib.utils.to_samples`)

    Example:
        >>> Time(0.5, 'seconds')(sampling_rate=8)
        4
        >>> Time(1000, 'ms')(sampling_rate=8)
        8
        >>> Time(16, 'samples')()
        16
        >>> Time(0.5, 'relative')(length=64)
        32
        >>> # generate randomized values
        >>> seed(0)
        >>> t = Time(observe.FloatUni(0.25, 0.75), 'relative')
        >>> t(length=64)
        33
        >>> t(length=64)
        38

    """
    def __init__(
            self,
            value: typing.Union[int, float, observe.Base],
            unit: str,
    ):

        self.value = value
        self.unit = unit.strip()

    def __call__(
            self,
            *,
            sampling_rate: int = None,
            length: int = None,
            allow_negative: bool = False,
    ) -> int:
        r"""Convert timestamp or timespan to number of samples.

        If ``unit`` is set to ``'samples'``,
        no argument must be given.
        In case of ``'relative'``,
        a value for ``length`` has to be provided.
        In any other case,
        a value for ``sampling_rate`` is required.

        Args:
            sampling_rate: sampling rate in Hz
            length: reference point if unit is ``relative``
                (in number of samples)
            allow_negative: allow negative values

        Returns:
            number of samples

        Raises:
            ValueError: if ``allow_negative`` is ``False``
                and computed value is negative
            ValueError: if time format is not supported
            ValueError: if ``length`` is not provided,
                but ``unit`` is ``'samples'``
            ValueError: if  ``sampling_rate`` is not provided,
                but ``unit`` is not ``'samples'`` or ``'relative'``

        """

        value = observe.observe(self.value)

        if self.unit == 'samples':
            num_samples = int(value)
        elif self.unit == 'relative':
            if length is None:
                raise ValueError(
                    "Unit is set to 'relative', "
                    "but no value is provided for 'length'."
                )
            num_samples = int(length * value)
        else:
            if sampling_rate is None:
                raise ValueError(
                    f"Unit is set to '{self.unit}', "
                    f"but no value is provided for 'sampling_rate'."
                )
            try:
                value = hf.parse_timespan(str(value) + self.unit)
                num_samples = int(value * sampling_rate)
            except hf.InvalidTimespan:
                raise ValueError(
                    f"Unknown time format "
                    f"'{self.unit}'."
                )

        if num_samples < 0 and not allow_negative:
            raise ValueError(
                f"Number of samples takes on negative value "
                f"'{num_samples}'."
                f"If this is expected, "
                f"set 'allow_negative=True' "
                f"to avoid this error.",
            )

        return num_samples
