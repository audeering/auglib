import pytest

import auglib


@pytest.mark.parametrize(
    'value, unit, length, sampling_rate, allow_negative, expected',
    [
        (1.0, 's', None, 8, False, 8),
        (-1.0, 's', None, 8, True, -8),
        (1000, 'ms', None, 8, False, 8),
        (-1000, 'ms', None, 8, True, -8),
        (8, 'samples', None, None, False, 8),
        (-8, 'samples', None, None, True, -8),
        (0.5, 'relative', 8, None, False, 4),
        (-0.5, 'relative', 8, None, True, -4),
        (2.0, 'relative', 8, None, False, 16),
        (-2.0, 'relative', 8, None, True, -16),
        pytest.param(  # sampling rate not set
            1.0, 's', None, None, None, False,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # length not set
            0.5, 'relative', None, None, None, False,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # invalid unit
            0.5, 'invalid', None, None, None, False,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # negative
            -0.5, 's', None, None, None, False,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_time(value, unit, length, sampling_rate, allow_negative, expected):
    assert expected == auglib.Time(value, unit)(
        length=length,
        sampling_rate=sampling_rate,
        allow_negative=allow_negative,
    )
