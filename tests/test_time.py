import pytest

import auglib


@pytest.mark.parametrize(
    'value, unit, length, sampling_rate, num_samples',
    [
        (1.0, 's', None, 8, 8),
        (1000, 'ms', None, 8, 8),
        (8, 'samples', None, None, 8),
        (0.5, 'relative', 8, None, 4),
        (2.0, 'relative', 8, None, 16),
        pytest.param(  # sampling rate not set
            1.0, 's', None, None, None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # length not set
            0.5, 'relative', None, None, None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_time(value, unit, length, sampling_rate, num_samples):
    assert num_samples == auglib.Time(value, unit)(
        length=length,
        sampling_rate=sampling_rate,
    )
