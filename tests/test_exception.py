import pytest

import auglib


def test_strategies():

    transform = auglib.transform.HighPass(8000)
    expected_msg = (
        'Exception from external C library: '
        'invalid cut-off frequency'
    )

    # default -> 'exception'
    with auglib.AudioBuffer(10, 8000, unit='samples') as buf:
        with pytest.raises(RuntimeError, match=expected_msg):
            transform(buf)

    # silently ignore error
    auglib.set_exception_handling('silent')
    with auglib.AudioBuffer(10, 8000, unit='samples') as buf:
        transform(buf)

    # only show a warning
    for strategy in ['warning', 'stacktrace']:
        auglib.set_exception_handling(strategy)
        with auglib.AudioBuffer(10, 8000, unit='samples') as buf:
            with pytest.warns(RuntimeWarning, match=expected_msg):
                transform(buf)

    # raise error
    auglib.set_exception_handling('exception')
    with auglib.AudioBuffer(10, 8000, unit='samples') as buf:
        with pytest.raises(RuntimeError, match=expected_msg):
            transform(buf)
