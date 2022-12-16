r"""Transform classes.

An augmentation pipeline is basically
a series of transformations.
Applying the transformations
to an audio signal
changes the sound and
the result is an augmented version
of the original signal.
:mod:`auglib.transform`
comes with a number of transformations,
which can be combined
using :class:`auglib.transform.Compose`, e.g.:

.. jupyter-execute::

    import auglib

    transform = auglib.transform.Compose(
        [
            auglib.transform.HighPass(cutoff=3000),
            auglib.transform.Clip(),
            auglib.transform.NormalizeByPeak(peak_db=-3),
        ]
    )

If we want to apply the augmentation,
we have to create an :class:`auglib.AudioBuffer` first.

.. jupyter-execute::

    sampling_rate = 8000
    with auglib.AudioBuffer(1.0, sampling_rate) as buf:
        transform(buf)  # apply transformation to buffer

Since :mod:`auglib` is just a wrapper
around a `C library`_,
this will create a C array and fill it with zeros.
The ``with`` statement ensures
that the C array gets freed afterwards.
We can use :class:`auglib.AudioBuffer.from_array()`
to initialize it from a :class:`numpy.ndarray` instead.

.. jupyter-execute::

    import numpy as np

    signal = np.ones([1, 1000])  # signal with shape (1, num_samples)
    with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)

Since ``transform(buf)``
is an in-place operation,
we can afterwards call ``buf.to_array()``
to get the augmented signal as a numpy array.

.. jupyter-execute::

    with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf:
        transform(buf)
        y = buf.to_array()  # get copy of augmented signal as numpy array

As explained in :ref:`usage`
with :class:`auglib.Augment`
we can simplify to:

.. jupyter-execute::

    augment = auglib.Augment(transform)
    y = augment(signal, sampling_rate)

Most transformations available
with :mod:`auglib.transform`
are implemented in the `C library`_.
However,
it is also possible to write
new transformations in pure Python.
In the following,
we will show two different ways
to implement a transformation that
repeats the current buffer n times.

One way is to implement
a new class that derives from
:class:`auglib.transform.Base`.

.. jupyter-execute::

    import numpy as np

    class Repeat(auglib.transform.Base):

        def __init__(
                self,
                repeats: int,
                *,
                bypass_prob: bool = False,
        ):
            super().__init__(bypass_prob)
            self.repeats = repeats

        def _call(self, buf: auglib.AudioBuffer):
            # get content of buffer as a numpy array
            # and use np.tile() to repeat it
            signal_org = buf.to_array(copy=False)
            signal_aug = np.tile(signal_org, self.repeats)
            # expand buffer to new size
            missing = signal_aug.shape[1] - signal_org.shape[1]
            auglib.transform.AppendValue(
                missing,
                unit='samples',
            )(buf)
            # copy augmented signal back to buffer
            buf._data[:] = signal_aug
            return buf

    transform = Repeat(3)

.. Store transfrom in different variable
.. so we can reuse it later
.. jupyter-execute::
    :hide-code:

    transform1 = transform

A much simpler approach is to use
:class:`auglib.transform.Function`.

.. jupyter-execute::

    def repeat(signal, sampling_rate, repeats):
        return np.tile(signal, repeats)

    transform = auglib.transform.Function(repeat, {'repeats': 3})


.. Ensure we get the same results
.. jupyter-execute::
    :hide-code:

    with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf1:
        transform1(buf1)
        with auglib.AudioBuffer.from_array(signal, sampling_rate) as buf2:
            transform(buf2)
            np.testing.assert_almost_equal(
                buf1._data,
                buf2._data,
                decimal=4,
            )
            assert len(buf1) == 3 * signal.size


.. _`C library`: https://gitlab.audeering.com/tools/auglib


"""
from auglib.core.transform import (
    Base,
    BabbleNoise,
    AMRNB,
    Append,
    AppendValue,
    BandPass,
    BandStop,
    Clip,
    ClipByRatio,
    Compose,
    CompressDynamicRange,
    Fade,
    FFTConvolve,
    Function,
    GainStage,
    HighPass,
    LowPass,
    Mask,
    Mix,
    NormalizeByPeak,
    PinkNoise,
    Prepend,
    PrependValue,
    Resample,
    Select,
    Shift,
    Tone,
    Trim,
    WhiteNoiseGaussian,
    WhiteNoiseUniform
)
