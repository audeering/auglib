===========================
Data Augmentation in Python
===========================

A Python wrapper for auglib_, a C++ library for audio data augmentation.

.. _auglib: https://gitlab.audeering.com/tools/auglib


Installation
============

To install ``auglib`` run:

.. code-block:: bash

    $ # Create and activate Python virtual environment, e.g.
    $ # virtualenv --python=python3 ${HOME}/.envs/auglib
    $ # source ${HOME}/.envs/auglib/bin/activate
    $ pip install auglib


Basic usage
===========

Create two buffers and mix them:

.. code-block:: python

    >>> from auglib import AudioBuffer
    >>> from auglib.transform import Mix
    >>> with AudioBuffer(5, 8000, value=1, unit='samples') as aux:
    >>>     with AudioBuffer(10, 8000, unit='samples') as base:
    >>>         Mix(aux)(base)
    array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])

Compose transformations (note the use of `FloatUni` and `IntUni` to randomize the outcome in every iteration):

.. code-block:: python

    >>> from auglib import AudioBuffer, FloatUni, IntUni
    >>> from auglib.transform import Mix
    >>> with AudioBuffer(0.5, 8000, 1.0) as aux:
    >>>     mix = Mix(aux, gain_aux_db=FloatUni(-6.0, 0.0),
    >>>               write_pos_base=IntUni(0, 2), unit='samples',
    >>>               bypass_prob=0.25)
    >>>    c = Compose([mix, Clip()])
    >>>    for i in range(10):
    >>>        with AudioBuffer(1.0, 8000) as base:
    >>>            c(base)
    [0.         0.89406085 0.89406085 ... 0.         0.         0.        ]
    [0.         0.55691683 0.55691683 ... 0.         0.         0.        ]
    [0.91738 0.91738 0.91738 ... 0.      0.      0.     ]
    [0.       0.951939 0.951939 ... 0.       0.       0.      ]
    [0.90289986 0.90289986 0.90289986 ... 0.         0.         0.        ]
    [0.         0.78088665 0.78088665 ... 0.         0.         0.        ]
    [0. 0. 0. ... 0. 0. 0.]
    [0.        0.9794204 0.9794204 ... 0.        0.        0.       ]
    [0.7016285 0.7016285 0.7016285 ... 0.        0.        0.       ]
    [0.        0.8946461 0.8946461 ... 0.        0.        0.       ]

Augment a list of files with random noise samples and playback the result:

.. code-block:: python

    >>> from auglib.utils import scan_files
    >>> from auglib.source import Read
    >>> from auglib.transform import Mix
    >>> # read speech files
    >>> root = './usage-example'
    >>> speech_files = list(scan_files(root, sub_dir='speech', pattern='*.wav',
    >>>                                full_path=True))
    >>> read = Read(StrList(speech_files))
    >>> # prepare augmentation with noise files
    >>> noise_files = list(scan_files(root, sub_dir='noise', pattern='*.wav',
    >>>                               full_path=True))
    >>> mix = Mix(aux=StrList(noise_files, draw=True), loop_aux=True)
    >>> # run augmentation and playback result
    >>> for _ in range(len(speech_files)):
    >>>     with read() as base:
    >>>         mix(base).play()

Save and load a transformation:

.. code-block:: python

    >>> from auglib.transform import Compose, BandPass, Clip
    >>> c = Compose([BandPass(center=500, bandwidth=1000),
                     Clip(threshold=-6.0, bypass_prob=0.5)])
    >>> c.dump('my')
    >>> print(Compose.load('my'))
    auglib.transform.Compose:
      bypass_prob: null
      transforms:
      - auglib.transform.BandPass:
          bypass_prob: null
          center: 500
          bandwidth: 1000
          order: 1
          design: butter
      - auglib.transform.Clip:
          bypass_prob: 0.5
          threshold: -6.0
          soft: false
          normalize: false


Documentation
=============

To build the documentation run the following commands:

.. code-block:: bash

    pip install -r docs/requirements.txt
    python setup.py build_sphinx

The generated files will be available in the directory ``build/sphinx/html/``


Tests
=====

To run the tests do:

.. code-block:: bash

    pip install -r tests/requirements.txt
    pytest tests/

