auglib.transform
================

.. automodule:: auglib.transform

:mod:`auglib.transform`
comes with a number of transforms,
that can augment an audio signal.

.. code-block:: python

    import auglib
    import numpy as np

    sampling_rate = 8000
    signal = np.ones((1, 4))
    transform = auglib.transform.GainStage(-10)
    transform(signal, sampling_rate)

Transforms can be combined
using :class:`auglib.transform.Compose`.

.. code-block:: python

    transform = auglib.transform.Compose(
        [
            auglib.transform.HighPass(cutoff=3000),
            auglib.transform.Clip(),
            auglib.transform.NormalizeByPeak(peak_db=-3),
        ]
    )

You can add your own Python based transform
using :class:`auglib.transform.Function`.

.. code-block:: python

    def repeat(signal, sampling_rate, repeats):
        import numpy as np
        return np.tile(signal, repeats)

    transform = auglib.transform.Function(repeat, {'repeats': 3})


.. autosummary::
    :toctree:
    :nosignatures:

    Base
    AMRNB
    Append
    AppendValue
    BabbleNoise
    BandPass
    BandStop
    Clip
    ClipByRatio
    Compose
    CompressDynamicRange
    Fade
    FFTConvolve
    Function
    GainStage
    HighPass
    LowPass
    Mask
    Mix
    NormalizeByPeak
    PinkNoise
    Prepend
    PrependValue
    Resample
    Select
    Shift
    Tone
    Trim
    WhiteNoiseGaussian
    WhiteNoiseUniform
