Examples
========

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


    # Ensure we can access public repo
    artifactory_user = os.environ.get('ARTIFACTORY_USERNAME')
    artifactory_key = os.environ.get('ARTIFACTORY_API_KEY')
    os.environ['ARTIFACTORY_USERNAME'] = 'anonymous'
    os.environ['ARTIFACTORY_API_KEY'] = ''

    audb.config.REPOSITORIES = [
        audb.Repository(
            name='data-public',
            host='https://audeering.jfrog.io/artifactory',
            backend='artifactory',
        ),
    ]

    blue = '#6649ff'
    green = '#55dbb1'

    sns.set(
        rc={
            'axes.facecolor': (0, 0, 0, 0),
            'figure.facecolor': (0, 0, 0, 0),
            'axes.grid' : False,
            'figure.figsize':(8, 3)
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

Here we will collect augmentation examples,
showing you
how to solve certain augmentation tasks
or how to
combine :mod:`auglib`
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
            auglib.transform.NormalizeByPeak(peak_db=-3),
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


.. Clean up

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # Restore artifactory user
    if artifactory_user is not None:
        os.environ['ARTIFACTORY_USERNAME'] = artifactory_user
    if artifactory_key is not None:
        os.environ['ARTIFACTORY_API_KEY'] = artifactory_key
