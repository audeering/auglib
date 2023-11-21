Installation
============

To install :mod:`auglib` run:

.. code-block:: bash

    $ # Create and activate Python virtual environment, e.g.
    $ # virtualenv --no-download --python=python3 ${HOME}/.envs/auglib
    $ # source ${HOME}/.envs/auglib/bin/activate
    $ pip install auglib

:class:`auglib.transform.AMRNB`,
requires ``ffmpeg``
with support for the AMR-NB format.
Under Ubuntu 22.04 this can be achieved by:

.. code-block:: bash

    $ sudo apt install ffmpeg libavcodec-extra58

On older or newer distributions,
the package might be named
``libavcodec-extra57``,
``libavcodec-extra59``,
and so on.

:class:`auglib.transform.CompressDynamicRange`
requires ``sox``,
which can be installed under Ubuntu by:

.. code-block::

    $ sudo apt install sox
