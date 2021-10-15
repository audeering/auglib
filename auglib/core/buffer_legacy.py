import audeer

from auglib.core import transform


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='auglib.transform.Base',
)
class Transform(transform.Base):
    pass
