.. _examples:

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
how to solve certain augmentation tasks.

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


.. _examples-artificial-reverb:

Artificial Reverb
-----------------

If you don't have enough examples of recorded reverb,
or want to tune one particular parameter of reverb,
you can artificially generate it.
Pedalboard_ provides you a reverb transform,
that let you adjust a bunch of parameters
in the range 0 to 1.
For more information on Pedalboard_
see the :ref:`Pedalboard section <external-pedalboard>`.
In the following,
we simply pick all parameters
randomly from a normal distribution.

.. jupyter-execute::

    auglib.utils.random_seed(0)

    def reverb(
            signal,
            sampling_rate,
            room_size,
            damping,
            wet_level,
            dry_level,
            width,
    ):
        r"""Reverb augmentation using pedalboard."""
        import pedalboard
        board = pedalboard.Pedalboard(
            [
                pedalboard.Reverb(
                    room_size=room_size,
                    damping=damping,
                    wet_level=wet_level,
                    dry_level=dry_level,
                    width=width,
                ),
            ],
            sample_rate=sampling_rate,
        )
        return board(signal)

    random_params = auglib.FloatNorm(mean=0.5, std=0.5, minimum=0, maximum=1)
    transform = auglib.transform.Compose(
        [
            auglib.transform.Function(
                reverb,
                function_args={
                    'room_size': random_params,
                    'damping': random_params,
                    'wet_level': random_params,
                    'dry_level': random_params,
                    'width': random_params,
                },
            ),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)
    
.. jupyter-execute::
    :hide-code:

    plot(signal_augmented, green, 'Artificial\nReverb')

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

.. _Pedalboard: https://github.com/spotify/pedalboard
