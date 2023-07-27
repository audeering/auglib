import audeer

import auglib.observe as observe


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.List',
)
class FloatList(observe.List):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.FloatNorm',
)
class FloatNorm(observe.FloatNorm):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.FloatNorm',
)
class FloatUni(observe.FloatUni):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.List',
)
class IntList(observe.List):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.IntUni',
)
class IntUni(observe.IntUni):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.List',
)
class StrList(observe.List):  # pragma: no cover
    pass
