import re

import numpy as np
import pytest

import audobject

import auglib


# Trim tests that should be independent of fill
@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("fill", ["none", "zeros", "loop"])
@pytest.mark.parametrize(
    "start_pos, end_pos, duration, unit, signal, expected",
    [
        (0, None, None, "samples", [1, 2, 3], [1, 2, 3]),
        (0, None, 0, "samples", [1, 2, 3], [1, 2, 3]),
        (0, None, 0, "seconds", [1, 2, 3], [1, 2, 3]),
        (0, None, 2, "samples", [1, 2, 3], [1, 2]),
        (1, None, 2, "samples", [1, 2, 3], [2, 3]),
        (None, 1, 1, "samples", [1, 2, 3], [2]),
    ],
)
def test_Trim(
    sampling_rate,
    fill,
    start_pos,
    end_pos,
    duration,
    unit,
    signal,
    expected,
):
    signal = np.array(signal)
    expected = np.array(
        expected,
        dtype=auglib.core.transform.DTYPE,
    )
    transform = auglib.transform.Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    np.testing.assert_array_equal(
        transform(signal, sampling_rate),
        expected,
        strict=True,
    )


# Trim, fill='none'
@pytest.mark.parametrize("unit", ["samples"])
@pytest.mark.parametrize("signal", [[1, 2, 3, 4]])
@pytest.mark.parametrize("fill", ["none"])
@pytest.mark.parametrize("fill_pos", ["right", "left", "both"])
@pytest.mark.parametrize(
    "start_pos, end_pos, duration, expected",
    [
        (None, None, None, [1, 2, 3, 4]),
        (None, None, 0, [1, 2, 3, 4]),
        (None, None, 2, [2, 3]),
        (None, None, 3, [1, 2, 3]),
        (None, None, 6, [1, 2, 3, 4]),
        (None, 2, None, [1, 2]),
        (None, 2, 0, [1, 2]),
        (None, 2, 3, [1, 2]),
        (0, None, None, [1, 2, 3, 4]),
        (0, None, 0, [1, 2, 3, 4]),
        (2, None, 3, [3, 4]),
        (0, 0, None, [1, 2, 3, 4]),
        (0, 2, None, [1, 2]),
        (2, 0, None, [3, 4]),
        (0, 0, 0, [1, 2, 3, 4]),
        (0, 2, 0, [1, 2]),
        (2, 0, 0, [3, 4]),
        (0, 2, 3, [1, 2]),
        (0, 2, 4, [1, 2]),
        (2, 0, 3, [3, 4]),
        (2, 0, 4, [3, 4]),
        (1, 1, 1, [2]),
        (1, 1, 3, [2, 3]),
    ],
)
def test_Trim_fill_none(
    unit,
    signal,
    fill,
    fill_pos,
    start_pos,
    end_pos,
    duration,
    expected,
):
    signal = np.array(signal)
    expected = np.array(
        expected,
        dtype=auglib.core.transform.DTYPE,
    )
    transform = auglib.transform.Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        fill_pos=fill_pos,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    np.testing.assert_array_equal(
        transform(signal),
        expected,
        strict=True,
    )


# Trim, fill='zeros'
@pytest.mark.parametrize("unit", ["samples"])
@pytest.mark.parametrize("signal", [[1, 2, 3, 4]])
@pytest.mark.parametrize("fill", ["zeros"])
@pytest.mark.parametrize(
    "start_pos, end_pos, duration, fill_pos, expected",
    [
        (None, None, 2, "right", [2, 3]),
        (None, None, 3, "right", [1, 2, 3]),
        (None, None, 6, "right", [1, 2, 3, 4, 0, 0]),
        (None, None, 6, "left", [0, 0, 1, 2, 3, 4]),
        (None, None, 6, "both", [0, 1, 2, 3, 4, 0]),
        (None, 2, 3, "right", [1, 2, 0]),
        (None, 2, 3, "left", [0, 1, 2]),
        (None, 2, 3, "both", [1, 2, 0]),
        (2, None, 3, "right", [3, 4, 0]),
        (2, None, 3, "left", [0, 3, 4]),
        (2, None, 3, "both", [3, 4, 0]),
        (2, None, 4, "right", [3, 4, 0, 0]),
        (2, None, 4, "left", [0, 0, 3, 4]),
        (2, None, 4, "both", [0, 3, 4, 0]),
        (0, 2, 3, "right", [1, 2, 0]),
        (0, 2, 3, "left", [0, 1, 2]),
        (0, 2, 3, "both", [1, 2, 0]),
        (0, 2, 4, "right", [1, 2, 0, 0]),
        (0, 2, 4, "left", [0, 0, 1, 2]),
        (0, 2, 4, "both", [0, 1, 2, 0]),
        (2, 0, 3, "right", [3, 4, 0]),
        (2, 0, 3, "left", [0, 3, 4]),
        (2, 0, 3, "both", [3, 4, 0]),
        (2, 0, 4, "right", [3, 4, 0, 0]),
        (2, 0, 4, "left", [0, 0, 3, 4]),
        (2, 0, 4, "both", [0, 3, 4, 0]),
        (1, 1, 3, "right", [2, 3, 0]),
        (1, 1, 3, "left", [0, 2, 3]),
        (1, 1, 3, "both", [2, 3, 0]),
    ],
)
def test_Trim_fill_zeros(
    unit,
    signal,
    fill,
    start_pos,
    end_pos,
    duration,
    fill_pos,
    expected,
):
    signal = np.array(signal)
    expected = np.array(
        expected,
        dtype=auglib.core.transform.DTYPE,
    )
    transform = auglib.transform.Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        fill_pos=fill_pos,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    np.testing.assert_array_equal(
        transform(signal),
        expected,
        strict=True,
    )


# Trim, fill='loop'
@pytest.mark.parametrize("unit", ["samples"])
@pytest.mark.parametrize("signal", [[1, 2, 3, 4]])
@pytest.mark.parametrize("fill", ["loop"])
@pytest.mark.parametrize(
    "start_pos, end_pos, duration, fill_pos, expected",
    [
        (None, None, 2, "right", [2, 3]),
        (None, None, 2, "left", [2, 3]),
        (None, None, 2, "both", [2, 3]),
        (None, None, 3, "right", [1, 2, 3]),
        (None, None, 3, "left", [1, 2, 3]),
        (None, None, 3, "both", [1, 2, 3]),
        (None, None, 6, "right", [1, 2, 3, 4, 1, 2]),
        (None, None, 6, "left", [3, 4, 1, 2, 3, 4]),
        (None, None, 6, "both", [4, 1, 2, 3, 4, 1]),
        (None, None, 8, "right", [1, 2, 3, 4, 1, 2, 3, 4]),
        (None, None, 8, "left", [1, 2, 3, 4, 1, 2, 3, 4]),
        (None, None, 8, "both", [3, 4, 1, 2, 3, 4, 1, 2]),
        (None, 2, 3, "right", [1, 2, 1]),
        (None, 2, 3, "left", [2, 1, 2]),
        (None, 2, 3, "both", [1, 2, 1]),
        (2, None, 3, "right", [3, 4, 3]),
        (2, None, 3, "left", [4, 3, 4]),
        (2, None, 3, "both", [3, 4, 3]),
        (2, None, 4, "right", [3, 4, 3, 4]),
        (2, None, 4, "left", [3, 4, 3, 4]),
        (2, None, 4, "both", [4, 3, 4, 3]),
        (0, 2, 3, "right", [1, 2, 1]),
        (0, 2, 3, "left", [2, 1, 2]),
        (0, 2, 3, "both", [1, 2, 1]),
        (0, 2, 4, "right", [1, 2, 1, 2]),
        (0, 2, 4, "left", [1, 2, 1, 2]),
        (0, 2, 4, "both", [2, 1, 2, 1]),
        (2, 0, 3, "right", [3, 4, 3]),
        (2, 0, 3, "left", [4, 3, 4]),
        (2, 0, 3, "both", [3, 4, 3]),
        (2, 0, 4, "right", [3, 4, 3, 4]),
        (2, 0, 4, "left", [3, 4, 3, 4]),
        (2, 0, 4, "both", [4, 3, 4, 3]),
        (1, 1, 3, "right", [2, 3, 2]),
        (1, 1, 3, "left", [3, 2, 3]),
        (1, 1, 3, "both", [2, 3, 2]),
        (None, 3, 6, "right", [1, 1, 1, 1, 1, 1]),
        (None, 3, 6, "left", [1, 1, 1, 1, 1, 1]),
        (None, 3, 6, "both", [1, 1, 1, 1, 1, 1]),
        (3, None, 6, "right", [4, 4, 4, 4, 4, 4]),
        (3, None, 6, "left", [4, 4, 4, 4, 4, 4]),
        (3, None, 6, "both", [4, 4, 4, 4, 4, 4]),
        (None, 2, 6, "right", [1, 2, 1, 2, 1, 2]),
        (None, 2, 6, "left", [1, 2, 1, 2, 1, 2]),
        (None, 2, 6, "both", [1, 2, 1, 2, 1, 2]),
        (2, None, 6, "right", [3, 4, 3, 4, 3, 4]),
        (2, None, 6, "left", [3, 4, 3, 4, 3, 4]),
        (2, None, 6, "both", [3, 4, 3, 4, 3, 4]),
    ],
)
def test_Trim_fill_loop(
    unit,
    signal,
    fill,
    start_pos,
    end_pos,
    duration,
    fill_pos,
    expected,
):
    signal = np.array(signal)
    expected = np.array(
        expected,
        dtype=auglib.core.transform.DTYPE,
    )
    transform = auglib.transform.Trim(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        fill=fill,
        fill_pos=fill_pos,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    np.testing.assert_array_equal(
        transform(signal),
        expected,
        strict=True,
    )


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("signal", [[1, 2, 3]])
@pytest.mark.parametrize(
    "start_pos, end_pos, duration, unit, error, error_msg",
    [
        (  # negative start_pos
            -1.0,
            None,
            None,
            "seconds",
            ValueError,
            "'start_pos' must be >=0.",
        ),
        (  # negative start_pos
            -1,
            None,
            None,
            "samples",
            ValueError,
            "'start_pos' must be >=0.",
        ),
        (  # negative end_pos
            None,
            -1.0,
            None,
            "seconds",
            ValueError,
            "'end_pos' must be >=0.",
        ),
        (  # negative end_pos
            None,
            -1,
            None,
            "samples",
            ValueError,
            "'end_pos' must be >=0.",
        ),
        (  # negative duration
            0,
            None,
            -1.0,
            "seconds",
            ValueError,
            "'duration' must be >=0.",
        ),
        (  # negative duration
            0,
            None,
            -1,
            "samples",
            ValueError,
            "'duration' must be >=0.",
        ),
        (  # duration too small
            0,
            None,
            0.0001,
            "seconds",
            ValueError,
            "Your combination of "
            "'duration' = 0.0001 seconds "
            "and 'sampling_rate' = 8000 Hz "
            "would lead to an empty signal "
            "which is forbidden.",
        ),
        (  # start_pos >= len(signal)
            3,
            None,
            None,
            "samples",
            ValueError,
            "'start_pos' must be <3.",
        ),
        (  # start_pos >= len(signal)
            3,
            None,
            4,
            "samples",
            ValueError,
            "'start_pos' must be <3.",
        ),
        (  # end_pos >= len(signal)
            None,
            3,
            None,
            "samples",
            ValueError,
            "'end_pos' must be <3.",
        ),
        (  # end_pos >= len(signal)
            None,
            3,
            4,
            "samples",
            ValueError,
            "'end_pos' must be <3.",
        ),
        (  # start_pos + end_pos >= len(signal)
            1,
            2,
            None,
            "samples",
            ValueError,
            "'start_pos' + 'end_pos' must be <3.",
        ),
        (  # start_pos + end_pos >= len(signal)
            1,
            2,
            4,
            "samples",
            ValueError,
            "'start_pos' + 'end_pos' must be <3.",
        ),
    ],
)
def test_Trim_error_call(
    sampling_rate,
    signal,
    start_pos,
    end_pos,
    duration,
    unit,
    error,
    error_msg,
):
    with pytest.raises(error, match=re.escape(error_msg)):
        transform = auglib.transform.Trim(
            start_pos=start_pos,
            end_pos=end_pos,
            duration=duration,
            unit=unit,
        )
        transform = audobject.from_yaml_s(
            transform.to_yaml_s(include_version=False),
        )

        signal = np.array(signal)
        transform(signal, sampling_rate)


@pytest.mark.parametrize(
    "fill, fill_pos, error, error_msg",
    [
        (  # wrong fill
            "unknown",
            "right",
            ValueError,
            (
                "Unknown fill strategy 'unknown'. "
                "Supported strategies are: "
                "none, zeros, loop."
            ),
        ),
        (  # wrong fill_pos
            "none",
            "unknown",
            ValueError,
            (
                "Unknown fill_pos 'unknown'. "
                "Supported positions are: "
                "right, left, both."
            ),
        ),
    ],
)
def test_Trim_error_init(
    fill,
    fill_pos,
    error,
    error_msg,
):
    with pytest.raises(error, match=re.escape(error_msg)):
        auglib.transform.Trim(
            fill=fill,
            fill_pos=fill_pos,
        )
