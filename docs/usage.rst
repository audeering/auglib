.. jupyter-execute::
    :hide-code:
    :hide-output:

    from IPython.display import Audio as play
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    import audb

    blue = '#6649ff'
    green = '#55dbb1'

    sns.set(
        rc={
            'axes.facecolor': (0, 0, 0, 0),
            'figure.facecolor': (0, 0, 0, 0),
            'axes.grid': False,
            'figure.figsize': (8, 2.5),
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
    
    
    def series_to_html(self):
        df = self.to_frame()
        df.columns = ['']
        return df._repr_html_()
    setattr(pd.Series, '_repr_html_', series_to_html)


    def index_to_html(self):
        return self.to_frame(index=False)._repr_html_()
    setattr(pd.Index, '_repr_html_', index_to_html)



Usage
=====

:mod:`auglib` lets you augment your audio data.
It provides a large collection of augmentations
in its sub-module :mod:`auglib.transform`,
which can be easily applied to a signal,
or whole database,
with :class:`auglib.Augment`.


.. jupyter-execute::

    import auglib

    transform = auglib.transform.Compose(
        [
            auglib.transform.HighPass(cutoff=5000),
            auglib.transform.Clip(),
            auglib.transform.NormalizeByPeak(peak_db=-3),
        ]
    )
    augment = auglib.Augment(transform)

Check how to include :ref:`external solutions <external>`
and look at the :ref:`examples <examples>`
for further inspiration.
Or continue reading here,
to see how to apply the augmentations
to different inputs.


Augment a signal
~~~~~~~~~~~~~~~~

We now load a signal from emodb_,
and apply our augmentation to it.

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
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    plot(signal, blue, 'Original\nAudio')

.. jupyter-execute::
    :hide-code:

    play(signal, rate=sampling_rate)

.. empty line for some extra space

|

.. jupyter-execute::
    :hide-code:

    plot(signal_augmented, green, 'Augmented\nAudio')

.. jupyter-execute::
    :hide-code:

    play(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

.. _emodb: http://data.pp.audeering.com/databases/emodb/emodb.html

  
Augment a database in memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`auglib.Augment` can apply the augmentation
to an audformat_ database.
To demonstrate this,
we load a subset of the emodb_ database
and augment it.

.. jupyter-execute::

    db = audb.load(
        'emodb',
        version='1.1.1',
        media=['wav/03a01Fa.wav', 'wav/03a01Nc.wav', 'wav/03a01Wa.wav'],
        verbose=False,
    )
    augment.process_index(db.files)

Generally, we can note that all :meth:`process_*` methods
return a column holding the augmented signals or segments.
However, this has two drawbacks.
Keeping results in memory may exceed available resources
for a large database.
And it may be expensive to redo the
augmentation every time we run an experiment.


Augment a database to disk
~~~~~~~~~~~~~~~~~~~~~~~~~~

Therefore, the interface offers another method
:meth:`auglib.Augment.augment`, which takes
as input an index, column or table conform to audformat_,
but instead of returning the augmented signals
it stores them back to disk.
The result is an index, column or table pointing to the augmented files.

.. jupyter-execute::

    augment.augment(data=db.files, cache_root='cache')

The files are stored inside the :file:`cache_root` folder,
and :meth:`auglib.Augment.augment`
will detect if the requested augmentation
is already in stored in cache,
or if it has to perform the augmentation.
If you don't specify :file:`cache_root`,
the default value of ``$HOME/auglib``
will be used.

If we pass a column instead of an index
the column data will be kept:

.. jupyter-execute::

    y = db['files']['speaker'].get()
    augment.augment(data=y, cache_root='cache')

Finally, we the repeat last command on a table,
this time keeping the original files
and augmenting every file twice.

.. jupyter-execute::

    table = db['files'].get()
    augment.augment(
        data=table,
        cache_root='cache',
        modified_only=False,
        num_variants=2,
    )

.. _audformat: https://audeering.github.io/audformat/data-format.html


Serialize
~~~~~~~~~

It's possible to serialize a
:class:`auglib.transform.Transform` object
to YAML.

.. jupyter-execute::

    print(transform.to_yaml_s())

And we can save it to a file and re-instantiate it from there.

.. jupyter-execute::

    import audobject

    file = 'transform.yaml'
    transform.to_yaml(file)
    transform_from_yaml = audobject.from_yaml(file)

We can prove that (with the same random seed)
the new object will give the same result.

.. jupyter-execute::

    import numpy as np

    augment_from_yaml = auglib.Augment(transform_from_yaml)
    signal_augmented_from_yaml = augment_from_yaml(signal, sampling_rate)

    np.testing.assert_equal(signal_augmented, signal_augmented_from_yaml)

.. Remove stored YAML file
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os
    os.remove(file)
