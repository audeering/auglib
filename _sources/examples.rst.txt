.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os

    from IPython.display import Audio
    import matplotlib.pyplot as plt
    import numpy as np

    import audb
    import audplot
    import auglib

    grey = "#5d6370"
    red = "#e13b41"


.. === Document starts here ===

.. _examples:

Examples
========

Augmentation examples,
showing you
how to solve certain augmentation tasks.

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


.. _examples-recorded-reverb:

Recorded Reverb
---------------

Recorded reverb can be used
to make machine learning models robust
against changes of the room.
In the following we
use binaural `impulse responses`_
recorded with a `dummy head`_
from the air_ dataset.
Its ``rir`` table holds recordings
for four different rooms
at different distances.

.. jupyter-execute::

    df = audb.load_table("air", "rir", version="1.4.2", verbose=False)
    set(df.room)

We load the left channel
of all impulse responses
stored in the ``rir`` table
and resample them to 16000 Hz.
We then randomly pick
an impulse response
during augmentation
with :class:`auglib.observe.List`.

.. jupyter-execute::

    auglib.seed(0)

    db = audb.load(
        "air",
        version="1.4.2",
        tables="rir",
        channels=[0],
        sampling_rate=16000,
        verbose=False,
    )
    transform = auglib.transform.Compose(
        [
            auglib.transform.FFTConvolve(
                auglib.observe.List(db.files, draw=True),
                keep_tail=False,
            ),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Recorded\nReverb")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

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

    auglib.seed(1)

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
        )
        return board(signal, sampling_rate)

    random_params = auglib.observe.FloatNorm(
        mean=0.5,
        std=0.5,
        minimum=0,
        maximum=1,
    )
    transform = auglib.transform.Compose(
        [
            auglib.transform.Function(
                reverb,
                function_args={
                    "room_size": random_params,
                    "damping": random_params,
                    "wet_level": random_params,
                    "dry_level": random_params,
                    "width": random_params,
                },
            ),
            auglib.transform.NormalizeByPeak(),
        ]
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Artificial\nReverb")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-music:

Music
-----

Music can be added
as a background signal
during training of a machine learning model.
We load a single music file from musan_
in this example.
We recommend to use all media files
from the ``music`` table,
when using the augmentation in a real application.
We randomly crop each music sample
with repetition,
attenuate it by -15 dB to -10 dB,
and add it to the original input signal.

.. jupyter-execute::

    auglib.seed(0)

    db = audb.load(
        "musan",
        tables="music",
        media="music/fma/music-fma-0097.wav",
        version="1.0.0",
        verbose=False,
    )

    transform = auglib.transform.Mix(
        auglib.observe.List(db.files, draw=True),
        gain_aux_db=auglib.observe.IntUni(-15, -10),
        read_pos_aux=auglib.observe.FloatUni(0, 1),
        unit="relative",
        loop_aux=True,
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Music")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-noise-snr:

Noise with fixed SNR
--------------------

When adding noise to a signal during augmentation,
it is often desired
to let the noise level
depend on the signal level
to achieve a fixed signal-to-noise (SNR)
between the two.

This can be achieved in :mod:`auglib`
by the ``snr_db`` argument.
The following example
adds pink noise
with a SNR of 10 dB
to the input signal.

.. jupyter-execute::

    auglib.seed(0)

    transform = auglib.transform.PinkNoise(snr_db=10)
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Pink\nNoise")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-band-pass-filtered-noise:

Band-Pass Filtered Noise
------------------------

Some augmentations
like :class:`auglib.transform.WhiteNoiseGaussian`
generate augmentation signals
that are added to the incoming signal.
Those generated augmentation signals
can be modified
with the help
of :class:`auglib.transform.Mix`
and its ``transform`` argument.

The following example
adds band-pass filtered white noise
to the input signal.

.. jupyter-execute::

    auglib.seed(0)

    transform = auglib.transform.Mix(
        auglib.transform.WhiteNoiseGaussian(),
        snr_db=15,
        transform=auglib.transform.BandPass(
            center=4000,
            bandwidth=1000,
        ),
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Band-Pass\nNoise")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-babble-noise:

Babble Noise
------------

Babble noise refers to having several speakers
in the background
all talking at the same time.
The easiest way to augment your signal
with babble noise
is to use another speech database.

In the next example, we use speech from musan_
and augment our signal with it
similar to Section 3.3
in `Snyder et al. 2018`_.
We only load 10 speech files from musan_
to speed the example up.
We recommend to use all media files,
when using the augmentation in a real application.

.. jupyter-execute::

    auglib.seed(1)

    db = audb.load(
        "musan",
        tables="speech",
        media=".*speech-librivox-000\d",
        version="1.0.0",
        verbose=False,
    )

    transform = auglib.transform.BabbleNoise(
        list(db.files),
        num_speakers=auglib.observe.IntUni(3, 7),
        snr_db=auglib.observe.IntUni(13, 20),
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Babble\nNoise")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


Telephone
---------

Telephone transmission is mainly characterised
by the applied transmission codec,
compare `Vu et al. 2019`_.
With :mod:`auglib` we can use
the Adaptive Multi-Rate audio codec
in its narrow band version (AMR-NB).
Here,
we select from three different codec bitrates,
and add the possibility of clipping
at the beginning,
and the possibility of additive noise
at the end of the processing.
The AMR-NB codec requires a sampling rate of 8000 Hz,
which :class:`auglib.Augment` can take care of.

.. jupyter-execute::

    auglib.seed(0)

    transform = auglib.transform.Compose(
        [
            auglib.transform.ClipByRatio(
                auglib.observe.FloatUni(0, 0.01),
                normalize=True,
            ),
            auglib.transform.AMRNB(
                auglib.observe.List([4750, 5900, 7400]),
            ),
            auglib.transform.WhiteNoiseGaussian(
                gain_db=auglib.observe.FloatUni(-35, -30),
                bypass_prob=0.7,
            ),
        ]
    )
    augment = auglib.Augment(
        transform,
        sampling_rate=8000,
        resample=True,
    )
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Telephone")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=8000)

.. empty line for some extra space

|


.. _examples-random-crop:

Random Crop
-----------

To target machine learning models
with a fixed signal input length,
random cropping of the signals
is often used.
The following example
uses :class:`auglib.transform.Trim`
to randomly crop the input to a length of 0.5 s.
If you are training with :mod:`torch`
and you want to apply the transform
during every epoch
you might consider
:class:`audtorch.transforms.RandomCrop` instead.

.. jupyter-execute::

    auglib.seed(0)

    transform = auglib.transform.Trim(
        start_pos=auglib.Time(auglib.observe.FloatUni(0, 1), unit="relative"),
        duration=0.5,
        fill="loop",
        unit="seconds",
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Random\nCrop")
    ax = plt.gca()
    _ = ax.set_xlim(0, signal.shape[0])

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-gated-noise:

Gated Noise
-----------

You might want to add temporarily changing background noise
to your signal.
The direct approach
is to simply switch the noise on and off
and generate gated background noise.
In the example,
we select a single noise file
from the ``noise`` table of musan_,
which includes 930 different files.
In a real application
you should augment with all of them.
A combination
of :class:`auglib.transform.Mask`
and :class:`auglib.transform.Mix`
reads the noise
starting from a random position,
and adds it every 0.5 s
to the target signal.

.. jupyter-execute::

    auglib.seed(0)

    db = audb.load(
        "musan",
        tables="noise",
        media="noise/free-sound/noise-free-sound-0003.wav",
        version="1.0.0",
        verbose=False,
    )

    transform = auglib.transform.Mask(
        auglib.transform.Mix(
            auglib.observe.List(db.files, draw=True),
            gain_aux_db=auglib.observe.IntUni(-15, 0),
            read_pos_aux=auglib.observe.FloatUni(0, 1),
            unit="relative",
            loop_aux=True,
        ),
        step=0.5,
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Gated\nNoise")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-pitch-shift:

Pitch Shift
-----------

You might want to change the pitch
of a speaker or singer
in your signal.
We use praat_ here
with the help of the :mod:`parselmouth` Python package.
To install it
you have to use the name ``praat-parselmouth``.
Internally,
it extracts the pitch contour,
changes the pitch,
and re-synthesises the audio signal.

.. jupyter-execute::

    import parselmouth
    from parselmouth.praat import call as praat

    auglib.seed(2)

    def pitch_shift(signal, sampling_rate, semitones):
        sound = parselmouth.Sound(signal, sampling_rate)
        manipulation = praat(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = praat(manipulation, "Extract pitch tier")
        factor = 2 ** (semitones / 12)
        praat(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
        praat([pitch_tier, manipulation], "Replace pitch tier")
        sound_transposed = praat(manipulation, "Get resynthesis (overlap-add)")
        return sound_transposed.values.flatten()

    transform = auglib.transform.Function(
        function=pitch_shift,
        function_args={"semitones": auglib.observe.IntUni(-4, 4)},
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Pitch\nShift")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. _examples-constant-pitch:

Constant Pitch
--------------

You might want to equalize the pitch
of the speakers in your database.
We use again praat_ to achieve this
as mentioned in :ref:`examples-pitch-shift`.

The first approach calculates the average pitch
of the input signal,
and adjusts it to the desired pitch given as f0 in Hz
by re-synthesizing the signal with a shifted pitch contour,
which preserves the natural pitch fluctuations per speaker.

.. jupyter-execute::

    import numpy as np
    import parselmouth
    from parselmouth.praat import call as praat

    def constant_pitch(signal, sampling_rate, desired_pitch):
        sound = parselmouth.Sound(signal, sampling_rate)
        # Estimate average pitch of signal
        pitch = sound.to_pitch()
        pitch = pitch.selected_array["frequency"]
        pitch[pitch == 0] = np.NaN
        pitch = np.nanmean(pitch)
        # Adjust signal to desired pitch
        manipulation = praat(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = praat(manipulation, "Extract pitch tier")
        factor = desired_pitch / pitch
        praat(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
        praat([pitch_tier, manipulation], "Replace pitch tier")
        sound_transposed = praat(manipulation, "Get resynthesis (overlap-add)")
        return sound_transposed.values.flatten()

    transform = auglib.transform.Function(
        function=constant_pitch,
        function_args={"desired_pitch": 100},
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Constant\nPitch")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|

The second approach specifies a constant pitch contour
representing the desired pitch,
which removes any pitch fluctuations from the signal
after re-synthesis.

.. jupyter-execute::

    import parselmouth
    from parselmouth.praat import call as praat

    def constant_pitch(signal, sampling_rate, desired_pitch):
        sound = parselmouth.Sound(signal, sampling_rate)
        manipulation = praat(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = praat(manipulation, "Create PitchTier", "Name", sound.xmin, sound.xmax)
        praat(pitch_tier, "Add point", sound.xmax / 2, desired_pitch)
        praat([pitch_tier, manipulation], "Replace pitch tier")
        sound_transposed = praat(manipulation, "Get resynthesis (overlap-add)")
        return sound_transposed.values.flatten()

    transform = auglib.transform.Function(
        function=constant_pitch,
        function_args={"desired_pitch": 100},
    )
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, sampling_rate)

.. jupyter-execute::
    :hide-code:

    audplot.waveform(signal_augmented, color=red, text="Constant\nPitch")

.. jupyter-execute::
    :hide-code:

    Audio(signal_augmented, rate=sampling_rate)

.. empty line for some extra space

|


.. === Links ===
.. _air: https://audeering.github.io/datasets/datasets/air.html
.. _dummy head: https://en.wikipedia.org/wiki/Dummy_head_recording
.. _impulse responses: https://en.wikipedia.org/wiki/Impulse_response
.. _musan: https://audeering.github.io/datasets/datasets/musan.html
.. _Pedalboard: https://github.com/spotify/pedalboard
.. _praat: https://github.com/praat/praat/
.. _Snyder et al. 2018: https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
.. _Vu et al. 2019: http://www.apsipa.org/proceedings/2019/pdfs/216.pdf
