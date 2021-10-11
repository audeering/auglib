.. _external:

External Solutions
==================

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os

    from IPython.display import Audio
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import audb
    import auglib


    blue = '#6649ff'
    green = '#55dbb1'

    sns.set(
        rc={
            'axes.facecolor': (0, 0, 0, 0),
            'figure.facecolor': (0, 0, 0, 0),
            'axes.grid' : False,
            'figure.figsize':(8, 2.5)
        },
    )


    def plot(signal, color, text):
        signal = np.atleast_2d(signal)[0, :]
        g = sns.lineplot(data=signal, color=color, linewidth=2.5)
        # Remove all axis
        sns.despine(left=True, bottom=True)
        g.tick_params(left=False, bottom=False)
        _ = g.set(xticklabels=[], yticklabels=[])
        _ = plt.xlim([-0.15 * len(signal), len(signal)])
        _ = plt.ylim([-1, 1])
        _ = plt.text(
            -0.02 * len(signal),
            0,
            text,
            fontsize='large',
            fontweight='semibold',
            color=color,
            horizontalalignment='right',
            verticalalignment='center',
        )


.. === Document starts here ===

Augmentation examples,
showing you
how to combine :mod:`auglib`
with external augmentation solutions.

Let's start with loading an example file to augment.

.. jupyter-execute::

    import audb
    import audiofile

    files = audb.load_media(
        'emodb',
        'wav/03a01Fa.wav',
        version='1.1.1',
        verbose=False,
    )
    signal, sampling_rate = audiofile.read(files[0], always_2d=True)

.. jupyter-execute::
    :hide-code:

    plot(signal, blue, 'Original\nAudio')

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

Pedalboard_ lacks direct documentation
of the parameters you can set for each of their classes.
You can list available parameters
by inspecting the attributes of a class:

.. jupyter-execute::

    import pedalboard

    [attr for attr in dir(pedalboard.Reverb) if not attr.startswith('_')]

For the value range
and an explanation
of the parameters,
you might want to look
at the corresponding documentation
of the underlying JUCE C code.
For reverb it is located at
https://docs.juce.com/master/structReverb_1_1Parameters.html

In the following example,
we use the compressor,
chorus,
phaser,
and reverb
augmentation from pedalboard_,
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
            sample_rate=sampling_rate,
        )
        return board(signal)

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

    plot(signal_augmented, green, 'Augmented\nAudio')

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
we combine gaussian noise,
time stretching,
and pitch shifting.
Similar to :mod:`auglib`
a probability controls if
a transformation is applied or bypassed.
Again,
we use the :class:`auglib.transform.Function` class
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
            auglib.transform.Function(audiomentations_transform, {'p': 1.0}),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    plot(signal_augmented, green, 'Augmented\nAudio')

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

.. _Audiomentations: https://github.com/iver56/audiomentations
.. _audiomentations: https://github.com/iver56/audiomentations
.. _albumentations: https://github.com/albumentations-team/albumentations
.. _torch-audiomentations: https://github.com/asteroid-team/torch-audiomentations
