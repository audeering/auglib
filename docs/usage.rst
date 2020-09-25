Usage
=====

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd


    def series_to_html(self):
        df = self.to_frame()
        df.columns = ['']
        return df._repr_html_()
    setattr(pd.Series, '_repr_html_', series_to_html)


    def index_to_html(self):
        return self.to_frame(index=False)._repr_html_()
    setattr(pd.Index, '_repr_html_', index_to_html)


    def plot(signal, sampling_rate):
        t = np.arange(len(signal)) / sampling_rate
        plt.xlabel('Seconds')
        plt.ylabel('Amplitude')
        plt.plot(t, signal)

.. jupyter-execute::

    import audb
    import audiofile
    import auglib

    auglib.utils.random_seed(1)

Transforms
~~~~~~~~~~

Define transforms to augment audio signals:

.. jupyter-execute::

    transform = auglib.transform.Compose([
        auglib.transform.AppendValue(5.0),
        auglib.transform.Append(
            'docs/_static/speech/sample1.wav',
            transform=auglib.transform.GainStage(4.0)
        ),
        auglib.transform.AppendValue(5.0),
        auglib.transform.Mix(
            'docs/_static/noise/babble.wav',
            gain_aux_db=2.0,
            read_pos_aux=20.0,
            transform=auglib.transform.NormalizeByPeak(),
        ),
        auglib.transform.Mix(
            'docs/_static/noise/music.wav',
            gain_aux_db=-12.0,
            write_pos_base=3.0,
            loop_aux=True,
            transform=auglib.transform.FFTConvolve(
                'docs/_static/ir/small-speaker.wav',
                keep_tail=True,
            )
        ),
        auglib.transform.FFTConvolve('docs/_static/ir/factory-hall.wav'),
        auglib.transform.BandPass(1000.0, 1800.0),
        auglib.transform.PinkNoise(gain_db=-30.0),
        auglib.transform.Clip(),
    ])

Apply transforms on a file and save the result.

.. jupyter-execute::

    with auglib.AudioBuffer.read('docs/_static/speech/sample2.wav') as buf:
        transform(buf)
        buf.write('docs/_static/speech/augmented.wav')

Listen to the original files:

.. raw:: html

    <audio controls="controls">
      <source src="_static/speech/sample1.wav" type="audio/wav">
      Your browser does not support the <code>audio</code> element.
    </audio>
    <br>
    <audio controls="controls">
      <source src="_static/speech/sample2.wav" type="audio/wav">
      Your browser does not support the <code>audio</code> element.
    </audio>
    <br>
    <br>

And the result:

.. raw:: html

    <audio controls="controls">
      <source src="_static/speech/augmented.wav" type="audio/wav">
      Your browser does not support the <code>audio</code> element.
    </audio>
    <br>
    <br>

Augment database
~~~~~~~~~~~~~~~~

In this section we will show how a :class:`auglib.Transform`
object can be applied to a database in `Unified Format`_.

Therefore, we pass it to an instance of :class:`auglib.Augment`,
which creates an interface with additional functions.

.. jupyter-execute::

    transform = auglib.transform.WhiteNoiseUniform()
    augment = auglib.Augment(
        transform=transform, # apply transformation
        sampling_rate=8000,  # and resample
        resample=True,       # to 8 kHz
        num_workers=5,       # using 5 threads
    )

To demonstrate the new interface,
we load a database with some empty files.

.. jupyter-execute::

    db = audb.load('testdata', version='1.5.0')

.. jupyter-execute::
    :hide-code:

    files = db.files[:5]
    signal, sampling_rate = audiofile.read(files[0])
    plot(signal, sampling_rate)

In memory
^^^^^^^^^

Through the interface, we can now apply the
transformation on a list of files.
The result is a column of augmented signals at 8 kHz.

.. jupyter-execute::

    result = augment.process_files(
        files=files,
    )
    result

If we plot one of the signals,
we see that they now contain noise.

.. jupyter-execute::
    :hide-code:

    plot(result[0][0], augment.sampling_rate)

We can do the same on a segmented index in the `Unified Format`_.

.. jupyter-execute::

    index = db.segments[:5]
    result = augment.process_unified_format_index(
        index=index,
    )
    result

The result is a column of augmented segments.
If we plot the first segment, we get:

.. jupyter-execute::
    :hide-code:

    plot(result[0][0], augment.sampling_rate)

Generally, we can note that all :meth:`process_*` methods
return a column holding the augmented signals or segments.
However, this has two drawbacks.
Keeping results in memory may exceed available resources
for a large database.
And it may be expensive to redo the
augmentation everytime we run an experiment.

To disk
^^^^^^^

Therefore, the interface offers another method
:meth:`auglib.Augment.augment`, which takes
as input a column or table in `Unified Format`_,
but instead of returning the augmented signals
it stores them back to disk.
The result is column or table with the same content (e.g. labels),
but a new index pointing to the augmented files.

.. jupyter-execute::

    column = db['happiness.dev.gold']['happiness'].get()[:10]
    result = augment.augment(
        column_or_table=column,
        cache_root='cache',
    )
    result

If we plot one of the augmented files,
we can spot the augmented segments.

.. jupyter-execute::
    :hide-code:

    augmented_signal, _ = audiofile.read(result.index[0][0])
    plot(augmented_signal, augment.sampling_rate)

Finally, we the repeat last command on a table,
this time keeping the original files
and augmenting every file twice.

.. jupyter-execute::

    table = db['happiness.dev.gold'].get()[:10]
    result = augment.augment(
        column_or_table=table,
        cache_root='cache',
        modified_only=False,
        num_variants=2,
    )
    result

.. _`Unified Format`: http://tools.pp.audeering.com/audata/data-tables.html
