import numpy as np
import pytest

from audmath import db
import audobject

import auglib


# Tests independent of window form
@pytest.mark.parametrize("signal", [np.array([1, 1, 1, 1, 1, 1])])
@pytest.mark.parametrize("unit", ["samples"])
@pytest.mark.parametrize(
    "in_shape",
    [
        "linear",
        "kaiser",
        "tukey",
        "exponential",
        "logarithmic",
    ],
)
@pytest.mark.parametrize(
    "out_shape",
    [
        "linear",
        "kaiser",
        "tukey",
        "exponential",
        "logarithmic",
    ],
)
@pytest.mark.parametrize(
    "in_dur, out_dur, in_db, out_db, expected",
    [
        (0, 0, -120, -120, np.array([1, 1, 1, 1, 1, 1])),
        (1, 0, -120, -120, np.array([0, 1, 1, 1, 1, 1])),
        (0, 1, -120, -120, np.array([1, 1, 1, 1, 1, 0])),
        (1, 1, -120, -120, np.array([0, 1, 1, 1, 1, 0])),
        (2, 2, -120, -120, np.array([0, 1, 1, 1, 1, 0])),
        (1, 0, db(0.5), db(0.3), np.array([0.5, 1, 1, 1, 1, 1])),
        (0, 1, db(0.5), db(0.3), np.array([1, 1, 1, 1, 1, 0.3])),
        (1, 1, db(0.5), db(0.3), np.array([0.5, 1, 1, 1, 1, 0.3])),
        (1, 1, db(0.5), -120, np.array([0.5, 1, 1, 1, 1, 0])),
        (1, 1, -120, db(0.3), np.array([0, 1, 1, 1, 1, 0.3])),
    ],
)
def test_Fade(
    signal,
    unit,
    in_shape,
    out_shape,
    in_dur,
    out_dur,
    in_db,
    out_db,
    expected,
):
    expected = expected.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.Fade(
        in_dur=in_dur,
        out_dur=out_dur,
        in_shape=in_shape,
        out_shape=out_shape,
        in_db=in_db,
        out_db=out_db,
        unit=unit,
    )
    augmented_signal = transform(signal)
    assert augmented_signal.dtype == expected.dtype
    assert augmented_signal.shape == expected.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected,
        decimal=4,
    )


@pytest.mark.parametrize("unit", ["samples"])
@pytest.mark.parametrize(
    "in_dur, out_dur, in_shape, out_shape, signal, expected",
    [
        # === Linear shape ===
        (
            0,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
        ),
        # Fade-in only
        (
            1,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0, 1, 1]),
        ),
        (
            2,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0, 1, 1]),
        ),
        (
            3,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0, 0.5, 1]),
        ),
        (
            4,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0, 0.3333, 0.6667]),
        ),
        (
            5,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0, 0.25, 0.5]),
        ),
        # Fade-out only
        (
            0,
            1,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([1, 1, 0]),
        ),
        (
            0,
            2,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([1, 1, 0]),
        ),
        (
            0,
            3,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([1, 0.5, 0]),
        ),
        (
            0,
            4,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0.6667, 0.3333, 0]),
        ),
        (
            0,
            5,
            "linear",
            "linear",
            np.array([1, 1, 1]),
            np.array([0.5, 0.25, 0]),
        ),
        # Fade-out + Fade-in on even signals
        (
            3,
            3,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.5, 1, 1, 0.5, 0]),
        ),
        (
            3,
            2,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.5, 1, 1, 1, 0]),
        ),
        (
            3,
            1,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.5, 1, 1, 1, 0]),
        ),
        (
            3,
            0,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.5, 1, 1, 1, 1]),
        ),
        (
            4,
            4,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.3333, 0.6667, 0.6667, 0.3333, 0]),
        ),
        (
            4,
            3,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.3333, 0.6667, 1, 0.5, 0]),
        ),
        # Fade-out + Fade-in on odd signals
        (
            4,
            4,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0.3333, 0.4444, 0.3333, 0]),
        ),
        (
            4,
            3,
            "linear",
            "linear",
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0.3333, 0.6667, 0.5, 0]),
        ),
        # === Other shapes ===
        (
            3,
            3,
            "kaiser",
            "kaiser",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([7.7269e-06, 4.6272e-01, 1, 1, 4.6272e-01, 7.7269e-06]),
        ),
        (
            3,
            3,
            "tukey",
            "tukey",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.5, 1, 1, 0.5, 0]),
        ),
        (
            3,
            3,
            "exponential",
            "exponential",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0.0, 0.26894142, 1.0, 1.0, 0.26894142, 0.0]),
        ),
        (
            3,
            3,
            "logarithmic",
            "logarithmic",
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0.63092975, 1, 1, 0.63092975, 0]),
        ),
    ],
)
def test_Fade_window_form(
    unit,
    in_dur,
    out_dur,
    in_shape,
    out_shape,
    signal,
    expected,
):
    expected = expected.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.Fade(
        in_dur=in_dur,
        out_dur=out_dur,
        in_shape=in_shape,
        out_shape=out_shape,
        unit=unit,
    )
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )
    augmented_signal = transform(signal)
    assert augmented_signal.dtype == expected.dtype
    assert augmented_signal.shape == expected.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected,
        decimal=4,
    )


@pytest.mark.parametrize(
    "in_shape, out_shape, in_db, out_db, error, error_msg",
    [
        (
            "some",
            "linear",
            -120,
            -120,
            ValueError,
            "Unknown fade shape 'some'.",
        ),
        (
            "linear",
            "some",
            -120,
            -120,
            ValueError,
            "Unknown fade shape 'some'.",
        ),
        (
            "linear",
            "linear",
            0,
            -120,
            ValueError,
            "Fading level needs to be below 0 dB, not 0 dB.",
        ),
        (
            "linear",
            "linear",
            -120,
            0,
            ValueError,
            "Fading level needs to be below 0 dB, not 0 dB.",
        ),
    ],
)
def test_Fade_error(
    in_shape,
    out_shape,
    in_db,
    out_db,
    error,
    error_msg,
):
    with pytest.raises(error, match=error_msg):
        auglib.transform.Fade(
            in_shape=in_shape,
            out_shape=out_shape,
            in_db=in_db,
            out_db=out_db,
        )
