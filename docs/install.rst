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
Under Ubuntu this can be achieved by:

.. code-block:: bash

    $ sudo apt install ffmpeg libavcodec-extra

:class:`auglib.transform.CompressDynamicRange`
requires ``sox``,
which can be installed under Ubuntu by:

.. code-block::

    $ sudo apt install sox
