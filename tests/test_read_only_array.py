import numpy as np
import pytest

import auglib


auglib.seed(0)


def identity(signal, sampling_rate):
    return signal


def read_only(
    signal: np.array,
    sampling_rate: int,
):
    signal.setflags(write=False)
    return signal


@pytest.mark.parametrize("signal", [[1, 1]])
@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize(
    "transform",
    [
        auglib.transform.AMRNB(4750),
        auglib.transform.Append(np.ones((1, 1))),
        auglib.transform.AppendValue(1, unit="samples"),
        auglib.transform.BabbleNoise([np.ones((1, 2))]),
        auglib.transform.BandPass(1000, 200),
        auglib.transform.BandStop(1000, 200),
        auglib.transform.Clip(),
        auglib.transform.ClipByRatio(0.05),
        auglib.transform.CompressDynamicRange(-15, 1 / 4),
        auglib.transform.Fade(in_dur=0.2, out_dur=0.7),
        auglib.transform.FFTConvolve(np.ones((1, 1))),
        auglib.transform.Function(identity),
        auglib.transform.GainStage(3),
        auglib.transform.HighPass(3000),
        auglib.transform.LowPass(100),
        auglib.transform.Mask(auglib.transform.Clip()),
        auglib.transform.Mix(np.ones((1, 1))),
        auglib.transform.NormalizeByPeak(),
        auglib.transform.PinkNoise(),
        auglib.transform.Prepend(np.ones((1, 1))),
        auglib.transform.PrependValue(1, unit="samples"),
        auglib.transform.Resample(4000),
        auglib.transform.Shift(1, unit="samples"),
        auglib.transform.Tone(100),
        auglib.transform.Trim(start_pos=0, end_pos=1, unit="samples"),
        auglib.transform.WhiteNoiseGaussian(),
        auglib.transform.WhiteNoiseUniform(),
    ],
)
def test_compose_read_only(
    signal: np.array,
    sampling_rate: int,
    transform: auglib.transform.Base,
):
    r"""Test applying transform on read-only array.

    Certain custom transforms
    (e.g. when using sox.Transformer)
    can return numpy arrays in read-only mode.

    If other transforms try to write to this array,
    without making a copy first,
    they will fail, see
    https://github.com/audeering/auglib/issues/31

    Args:
        signal: signal
        sampling_rate: sampling rate in Hz
        transform: transform

    """
    signal = np.array(signal, dtype=auglib.core.transform.DTYPE)

    # Apply transform to read-only signal
    signal.setflags(write=False)
    augmented_signal = transform(signal, sampling_rate)
    assert augmented_signal.flags["WRITEABLE"]

    # Apply transform in compose
    # after transform that makes signal read-only
    compose_transform = auglib.transform.Compose(
        [auglib.transform.Function(read_only), transform]
    )
    augmented_signal = compose_transform(signal, sampling_rate)
    assert augmented_signal.flags["WRITEABLE"]
