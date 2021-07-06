Usage
=====

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    import audb


    audb.config.REPOSITORIES = [
        audb.Repository(
            name='data-public-local',
            host='https://artifactory.audeering.com/artifactory',
            backend='artifactory',
        ),
    ]


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

Serialize
~~~~~~~~~

It's possible to serialize a
:class:`auglib.Transform` object
to YAML.

.. jupyter-execute::

    print(transform.to_yaml_s())

And we can save it to a file and re-instantiate it from there.

.. jupyter-execute::

    file = 'transform.yaml'
    transform.to_yaml(file)
    transform_from_yaml = auglib.Transform.from_yaml(file)

We can prove that (with the same random seed)
the new object will give the same result.

.. jupyter-execute::

    import numpy as np

    auglib.utils.random_seed(1)
    with auglib.AudioBuffer.read('docs/_static/speech/sample2.wav') as buf:
        y = transform(buf).data.copy()

    auglib.utils.random_seed(1)
    with auglib.AudioBuffer.read('docs/_static/speech/sample2.wav') as buf:
        y_from_yaml = transform_from_yaml(buf).data.copy()

    np.testing.assert_equal(y, y_from_yaml)

Augment database
~~~~~~~~~~~~~~~~

In this section we will show how a :class:`auglib.Transform`
object can be applied to a audformat_ database.
Therefore, we pass it to an instance of :class:`auglib.Augment`.

.. jupyter-execute::

    transform = auglib.transform.WhiteNoiseUniform()
    augment = auglib.Augment(
        transform=transform, # apply transformation
        num_workers=5,       # using 5 threads
    )

To demonstrate the :class:`auglib.Augment` interface,
we load a database with some empty files.

.. Pre-load database to cache
.. jupyter-execute::
    :stderr:
    :hide-code:
    :hide-output:

    db = audb.load('testdata', version='1.5.0')

.. jupyter-execute::

    db = audb.load('testdata', version='1.5.0')

.. jupyter-execute::
    :hide-code:

    signal, sampling_rate = audiofile.read(db.files[0])
    plot(signal, sampling_rate)

In memory
^^^^^^^^^

Through the interface, we can now apply the
transformation on a list of files.
The result is a column of augmented signals at 8 kHz.

.. jupyter-execute::

    files = db.files[:5]
    result = augment.process_files(
        files=files,
    )
    result

If we plot one of the signals,
we see that they now contain noise.

.. jupyter-execute::
    :hide-code:

    plot(result[0][0], sampling_rate)

We can do the same on a segmented index conform to audformat_.

.. jupyter-execute::

    index = db.segments[:5]
    result = augment.process_index(
        index=index,
    )
    result

The result is a column of augmented segments.
If we plot the first segment, we get:

.. jupyter-execute::
    :hide-code:

    plot(result[0][0], sampling_rate)

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
as input an index, column or table conform to audformat_,
but instead of returning the augmented signals
it stores them back to disk.
The result is an index, column or table pointing to the augmented files.

.. jupyter-execute::

    segments = db.segments[:10]
    result = augment.augment(
        data=segments,
        cache_root='cache',
    )
    result

If we plot one of the augmented files,
we can spot the augmented segments.

.. jupyter-execute::
    :hide-code:

    augmented_signal, _ = audiofile.read(result[0][0])
    plot(augmented_signal, sampling_rate)

Instead of an index, we also pass a column and
the column data will be kept:

.. jupyter-execute::

    column = db['happiness.dev.gold']['happiness'].get()[:10]
    result = augment.augment(
        data=column,
        cache_root='cache',
    )
    result

Finally, we the repeat last command on a table,
this time keeping the original files
and augmenting every file twice.

.. jupyter-execute::

    table = db['happiness.dev.gold'].get()[:10]
    result = augment.augment(
        data=table,
        cache_root='cache',
        modified_only=False,
        num_variants=2,
    )
    result

.. _audformat: https://audeering.github.io/audformat/data-format.html
