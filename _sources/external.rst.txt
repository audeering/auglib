.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os

    from IPython.display import Audio
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import audb
    import audplot
    import auglib


    grey = "#5d6370"
    red = "#e13b41"


.. === Document starts here ===

.. _external:

External Solutions
==================

Augmentation examples,
showing you
how to combine :mod:`auglib`
with external augmentation solutions.

Let's start with loading an example file to augment.

.. jupyter-execute::

    import audb
    import audiofile

    files = audb.load_media(
        "emodb",
        "wav/03a01Fa.wav",
        version="1.4.1",
        verbose=False,
    )
    signal, sampling_rate = audiofile.read(files[0])

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal, color=grey, text="Original\nAudio")

.. jupyter-execute::
    :hide-code:

    Audio(signal, rate=sampling_rate)

.. empty line for some extra space

|


.. _external-pedalboard:

Pedalboard
----------

Pedalboard_ is a Python package from Spotify,
that provides a collection
of useful and fast augmentations.
It also allows you
to include VST plugins
in your augmentation pipeline.

The documentation of Pedalboard_
does not discuss all the used parameters
of the augmentations.
For the value range
and an explanation
of the parameters,
you might want to look
at the corresponding documentation
of the underlying JUCE C code.
E.g. for :class:`pedalboard.Reverb`
it is located at
https://docs.juce.com/master/structReverb_1_1Parameters.html

In the following example,
we use the compressor,
chorus,
phaser,
and reverb
from pedalboard_,
as part of our :mod:`auglib`
augmentation chain
with the help of the :class:`auglib.transform.Function` class.

.. jupyter-execute::

    def pedalboard_transform(signal, sampling_rate):
        r"""Custom augmentation using pedalboard."""
        import pedalboard
        board = pedalboard.Pedalboard(
            [
                pedalboard.Compressor(threshold_db=-50, ratio=25),
                pedalboard.Chorus(),
                pedalboard.Phaser(),
                pedalboard.Reverb(room_size=0.25),
            ],
        )
        return board(signal, sampling_rate)

    transform = auglib.transform.Compose(
        [
            auglib.transform.Function(pedalboard_transform),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)
    
.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Augmented\nAudio")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

.. _Pedalboard: https://github.com/spotify/pedalboard
.. _pedalboard: https://github.com/spotify/pedalboard


.. _external-audiomentations:

Audiomentations
---------------

Audiomentations_ is another Python library
for audio data augmentation,
originally inspired by albumentations_.
It provides additional transformations
such as pitch shifting and time stretching,
or mp3 compression to
simulate lower audio quality.
It also includes spectrogram transformations
(not supported by :mod:`auglib`).
For GPU support the package
torch-audiomentations_
is available.

In the following example,
we combine Gaussian noise,
time stretching,
and pitch shifting.
Similar to :mod:`auglib`
a probability controls if
a transformation is applied or bypassed.
Again,
we use :class:`auglib.transform.Function`
to include transforms from audiomentations_
into our :mod:`auglib` augmentation chain.

.. jupyter-execute::

    def audiomentations_transform(signal, sampling_rate, p):
        r"""Custom augmentation using audiomentations."""
        import audiomentations
        compose = audiomentations.Compose([
            audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p),
            audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=p),
            audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=p),
        ])
        return compose(signal, sampling_rate)

    transform = auglib.transform.Compose(
        [
            auglib.transform.Function(audiomentations_transform, {"p": 1.0}),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Augmented\nAudio")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

.. _Audiomentations: https://github.com/iver56/audiomentations
.. _audiomentations: https://github.com/iver56/audiomentations
.. _albumentations: https://github.com/albumentations-team/albumentations
.. _torch-audiomentations: https://github.com/asteroid-team/torch-audiomentations


.. _external-sox:

Sox
---

Sox_ provides a large variety of effects,
so called Transformers_,
that might be useful for augmentation.
Here,
we shift the pitch by two semitones,
and apply a `Flanger effect`_.

.. jupyter-execute::

    def sox_transform(signal, sampling_rate):
        r"""Custom augmentation using sox."""
        import sox
        tfm = sox.Transformer()
        tfm.pitch(2)
        tfm.flanger()
        return tfm.build_array(
            input_array=signal.squeeze(),
            sample_rate_in=sampling_rate,
        )

    transform = auglib.transform.Compose(
        [
            auglib.transform.Function(sox_transform),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Augmented\nAudio")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

.. _Sox: https://pysox.readthedocs.io/en/latest/
.. _Transformers: https://pysox.readthedocs.io/en/latest/api.html#module-sox.transform
.. _Flanger effect: https://en.wikipedia.org/wiki/Flanging
