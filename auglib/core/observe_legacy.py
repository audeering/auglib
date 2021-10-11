import auglib.observe as observe

import audeer


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.FloatList',
)
class FloatList(observe.FloatList):  # pragma: no cover
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
    alternative='observe.IntList',
)
class IntList(observe.IntList):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.IntUni',
)
class IntUni(observe.IntUni):  # pragma: no cover
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='observe.StrList',
)
class StrList(observe.StrList):  # pragma: no cover
    pass
