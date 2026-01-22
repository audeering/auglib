.. invisible-code-block: python

    import audb
    import audiofile
    import auglib

    transform = auglib.transform.Compose(
        [
            auglib.transform.HighPass(cutoff=5000),
            auglib.transform.Clip(),
            auglib.transform.NormalizeByPeak(peak_db=-3),
        ]
    )
    augment = auglib.Augment(transform)

    files = audb.load_media(
        "emodb",
        "wav/03a01Fa.wav",
        version="1.4.1",
        verbose=False,
    )
    signal, sampling_rate = audiofile.read(files[0])

.. plot::
    :context: close-figs

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    import audb
    import audplot

    grey = "#5d6370"
    red = "#e13b41"

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


.. plot::
    :context: close-figs

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

.. plot::
    :context: close-figs
    :include-source:

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

.. plot::
    :context: close-figs

    audplot.waveform(signal, color=grey, text="Original\nAudio")

.. plot::
    :context: close-figs

    audiofile.write(
        audeer.path(static_dir, "usage-original0.wav"),
        signal,
        sampling_rate,
    )

.. raw:: html

    <p style="margin-left: 24px;">
        <audio controls src="media/usage-original0.wav"></audio>
    </p>


.. plot::
    :context: close-figs

    audplot.waveform(signal_augmented, color=red, text="Augmented\nAudio")

.. plot::
    :context: close-figs

    audiofile.write(
    audeer.path(static_dir, "usage-augmented0.wav"),
        signal_augmented,
        sampling_rate,
    )

.. raw:: html

    <p style="margin-left: 24px;">
        <audio controls src="media/usage-augmented0.wav"></audio>
    </p>



Augment files in memory
~~~~~~~~~~~~~~~~~~~~~~~

:class:`auglib.Augment` can apply the augmentation
to a list of files.
We load three files from emodb_,
and augment them using :meth:`auglib.Augment.process_files`.

>>> files = audb.load_media(
...     "emodb",
...     ["wav/03a01Fa.wav", "wav/03a01Nc.wav", "wav/03a01Wa.wav"],
...     version="1.4.1",
...     verbose=False,
... )
>>> augment.process_files(files)
file                        start   end
...emodb...03a01Fa.wav  0 days  0 days 00:00:01.898250       [[...
...emodb...03a01Nc.wav  0 days  0 days 00:00:01.611250       [[...
...emodb...03a01Wa.wav  0 days  0 days 00:00:01.877812500    [[...
dtype: object

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

>>> db = audb.load(
...     "emodb",
...     version="1.4.1",
...     media=["wav/03a01Fa.wav", "wav/03a01Nc.wav", "wav/03a01Wa.wav"],
...     verbose=False,
... )
>>> index = db.files
>>> augment.augment(index, cache_root="cache")
MultiIndex([('...cache...03a01Fa.wav', ...),
            ('...cache...03a01Nc.wav', ...),
            ('...cache...03a01Wa.wav', ...)],
           names=['file', 'start', 'end'])

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

>>> y = db["files"]["speaker"].get()
>>> augment.augment(y, cache_root="cache")
file                        start   end
...cache...03a01Fa.wav  0 days  0 days 00:00:01.898250       3
...cache...03a01Nc.wav  0 days  0 days 00:00:01.611250       3
...cache...03a01Wa.wav  0 days  0 days 00:00:01.877812500    3
Name: speaker, dtype: ...
Categories ...

Finally,
we augment a dataframe,
this time keeping the original files in the result
and augmenting every file twice.

>>> df = db["files"].get()
>>> df_augmented = augment.augment(
...     df,
...     cache_root="cache",
...     modified_only=False,
...     num_variants=2,
... )
>>> len(df_augmented)
9


Serialize
~~~~~~~~~

It's possible to serialize a
:class:`auglib.Augment` object
to YAML.

>>> print(augment.to_yaml_s())
$auglib.core.interface.Augment...:
  transform:
    $auglib.core.transform.Compose...:
      transforms:
      - $auglib.core.transform.HighPass...:
          cutoff: 5000
          order: 1
          design: butter
          preserve_level: false
          bypass_prob: null
      - $auglib.core.transform.Clip...:
          threshold: 0.0
          soft: false
          normalize: false
          preserve_level: false
          bypass_prob: null
      - $auglib.core.transform.NormalizeByPeak...:
          peak_db: -3
          clip: false
          preserve_level: false
          bypass_prob: null
      preserve_level: false
      bypass_prob: null
  sampling_rate: null
  resample: false
  channels: null
  mixdown: false
  seed: null
<BLANKLINE>

We can save it to a file
and re-instantiate it from there.

>>> import audobject
>>> file = "transform.yaml"
>>> augment.to_yaml(file)
>>> augment_from_yaml = audobject.from_yaml(file)
>>> augment_from_yaml(signal, sampling_rate)
array([[ 1.2283714e-03,  4.1666315e-03, -1.8338256e-03, ...,
        -4.1018693e-06,  8.3834183e-04, -1.6675657e-04]], shape=(1, 30372), dtype=float32)

The new object creates the exact same augmentation.
To make an augmentation reproducible
that includes random behavior
we have to set the ``seed`` argument.

>>> transform = auglib.transform.PinkNoise(gain_db=-5)
>>> augment = auglib.Augment(transform, seed=0)
>>> augment(signal, sampling_rate)
array([[0.2510293 , 0.19091995, 0.23076648, ..., 0.12397236, 0.15249531,
        0.16985646]], shape=(1, 30372), dtype=float32)

When we serialize the object,
the seed will be stored to YAML
and used to re-initialize the
random number generator when
the object is loaded.

>>> augment.to_yaml(file)
>>> augment_from_yaml = audobject.from_yaml(file)
>>> augment_from_yaml(signal, sampling_rate)
array([[0.2510293 , 0.19091995, 0.23076648, ..., 0.12397236, 0.15249531,
        0.16985646]], shape=(1, 30372), dtype=float32)

If we wanted a different random seed
we can also overwrite the value.

>>> augment_other_seed = audobject.from_yaml(file, override_args={"seed": 1})
>>> augment_other_seed(signal, sampling_rate)
array([[-0.01383871, -0.10363714, -0.12082221, ..., -0.21219613,
        -0.08782648, -0.14412443]], shape=(1, 30372), dtype=float32)


.. _emodb: https://audeering.github.io/datasets/datasets/emodb.html
.. _filewise index: https://audeering.github.io/audformat/data-tables.html#filewise
.. _segmented index: https://audeering.github.io/audformat/data-tables.html#segmented
