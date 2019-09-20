===========================
Data Augmentation in Python
===========================

A Python wrapper for auglib_, a C++ library for audio data augmentation.

.. _auglib: http://gitlab2.audeering.local/tools/auglib


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

.. code-block:: Python

    >>> from auglib import AudioBuffer
    >>> base = AudioBuffer(10, 8000)
    >>> aux = AudioBuffer(5, 8000)
    >>> aux.data += 1
    >>> base.mix(aux)
    >>> base
    array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.], dtype=float32)


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

