.. jupyter-execute::
    :hide-code:
    :hide-output:

    from IPython.display import Audio as play
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    import audb
    import audplot

    grey = "#5d6370"
    red = "#e13b41"

    
    def series_to_html(self):
        df = self.to_frame()
        df.columns = [""]
        return df._repr_html_()
    setattr(pd.Series, "_repr_html_", series_to_html)


    def index_to_html(self):
        return self.to_frame(index=False)._repr_html_()
    setattr(pd.Index, "_repr_html_", index_to_html)


.. === Document starts here ===

.. _usage:

Usage
=====

:mod:`auglib` lets you augment audio data.
It provides transformations
in its sub-module :mod:`auglib.transform`.
:class:`auglib.Augment` can then apply
those transformations
to a signal,
a file,
or a whole dataset.

Check how to include :ref:`external solutions <external>`
and look at the :ref:`examples <examples>`
for further inspiration.
Or continue reading here,
to see how to apply the augmentations
to different inputs.


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


Augment a signal
~~~~~~~~~~~~~~~~

We now load a signal from emodb_,
and apply our augmentation to it.

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
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal, color=grey, text="Original\nAudio")

.. jupyter-execute::
    :hide-code:

    play(signal, rate=sampling_rate)

.. empty line for some extra space

|

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Augmented\nAudio")

.. jupyter-execute::
    :hide-code:

    play(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

  
Augment files in memory
~~~~~~~~~~~~~~~~~~~~~~~

:class:`auglib.Augment` can apply the augmentation
to a list of files.
We load three files from emodb_,
and augment them using :meth:`auglib.Augment.process_files`.

.. jupyter-execute::

    files = audb.load_media(
        "emodb",
        ["wav/03a01Fa.wav", "wav/03a01Nc.wav", "wav/03a01Wa.wav"],
        version="1.4.1",
        verbose=False,
    )
    y_augmented = augment.process_files(files)
    y_augmented

All :meth:`process_*` methods
return a series
(:class:`pd.Series`)
holding the augmented signals
with a :class:`pd.MultiIndex`
containing the levels ``file``,
``start``,
``end``
(`segmented index`_).


Augment a dataset to disk
~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`auglib.Augment.augment` augments
a dataset to a cache folder on disk.
It takes as input an index,
series or dataframe.
The index needs at least one level,
named ``file``
holding file paths
(`filewise index`_)
or the three levels ``file``,
``start``,
and ``end``,
holding information on start and end
for segments of the file
provided as :class:`pd.Timedelta`
(`segmented index`_).
:meth:`auglib.Augment.augment`
returns an index, series, or table
with a `segmented index`_
that points to the augmented files.

The next example
loads the emodb_ dataset,
limited to three files for this example.
It then uses :class:`audformat.Database.files`
to get a `filewise index`_
pointing to the files of the dataset.

.. jupyter-execute::

    db = audb.load(
        "emodb",
        version="1.4.1",
        media=["wav/03a01Fa.wav", "wav/03a01Nc.wav", "wav/03a01Wa.wav"],
        verbose=False,
    )
    index = db.files
    index_augmented = augment.augment(index, cache_root="cache")
    index_augmented

The augmented files are stored inside the ``cache_root`` folder.
If :meth:`auglib.Augment.augment`
is called again on the same index,
it detects the requested augmentation
in cache,
and returns directly its result.
If you don't specify ``cache_root``,
the default value of ``$HOME/auglib``
will be used.

If we pass a series instead of an index
a series will be returned:

.. jupyter-execute::

    y = db["files"]["speaker"].get()
    y_augmented = augment.augment(y, cache_root="cache")
    y_augmented

Finally,
we augment a dataframe,
this time keeping the original files in the result
and augmenting every file twice.

.. jupyter-execute::

    df = db["files"].get()
    df_augmented = augment.augment(
        df,
        cache_root="cache",
        modified_only=False,
        num_variants=2,
    )
    df_augmented


Serialize
~~~~~~~~~

It's possible to serialize a
:class:`auglib.Augment` object
to YAML.

.. jupyter-execute::

    print(augment.to_yaml_s())

We can save it to a file
and re-instantiate it from there.

.. jupyter-execute::

    import audobject

    file = "transform.yaml"
    augment.to_yaml(file)
    augment_from_yaml = audobject.from_yaml(file)
    augment_from_yaml(signal, sampling_rate)

The new object creates the exact same augmentation.
To make an augmentation reproducible
that includes random behavior
we have to set the ``seed`` argument.

.. jupyter-execute::

    transform = auglib.transform.PinkNoise(gain_db=-5)
    augment = auglib.Augment(transform, seed=0)
    augment(signal, sampling_rate)

When we serialize the object,
the seed will be stored to YAML
and used to re-initialize the
random number generator when
the object is loaded.

.. jupyter-execute::

    augment.to_yaml(file)
    augment_from_yaml = audobject.from_yaml(file)
    augment_from_yaml(signal, sampling_rate)

If we wanted a different random seed
we can also overwrite the value.

.. jupyter-execute::

    augment_other_seed = audobject.from_yaml(file, override_args={"seed": 1})
    augment_other_seed(signal, sampling_rate)


.. Remove stored YAML file
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os
    os.remove(file)


.. _emodb: https://audeering.github.io/datasets/datasets/emodb.html
.. _filewise index: https://audeering.github.io/audformat/data-tables.html#filewise
.. _segmented index: https://audeering.github.io/audformat/data-tables.html#segmented
