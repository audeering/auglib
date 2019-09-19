import pytest
import numpy as np

from auglib import Tone, ToneShape


@pytest.mark.parametrize('freq', [1, 440])
def test_sine(freq):

    sr = 8000
    n = sr

    tone = Tone(n, sr, shape=ToneShape.SINE, freq=freq, unit='samples')
    sine = np.sin((np.arange(n, dtype=np.float) / sr) * 2 * np.pi * freq)
    np.testing.assert_almost_equal(tone.data, sine, decimal=3)
    tone.free()
