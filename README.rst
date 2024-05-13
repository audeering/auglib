======
auglib
======

|tests| |coverage| |docs| |python-versions| |license|

**auglib** is an augmentation library,
which provides transforms_
to modify audio signals_ and files_.

.. code-block:: python

    import auglib

    transform = auglib.transform.PinkNoise()
    augment = auglib.Augment(transform)
    augment(signal, sampling_rate)

**auglib** can further augment whole datasets_
while caching the results.
It can be easily extended
with external_ or `user defined`_ augmentations.

Have a look at the installation_ instructions
and listen to examples_.


.. _datasets: https://audeering.github.io/auglib/usage.html#augment-a-dataset-to-disk
.. _examples: https://audeering.github.io/auglib/examples.html
.. _external: https://audeering.github.io/auglib/external.html
.. _files: https://audeering.github.io/auglib/usage.html#augment-files-in-memory
.. _installation: https://audeering.github.io/auglib/install.html
.. _signals: https://audeering.github.io/auglib/usage.html#augment-a-signal
.. _transforms: https://audeering.github.io/auglib/api/auglib.transform.html
.. _usage: https://audeering.github.io/auglib/usage.html
.. _user defined: https://audeering.github.io/auglib/api/auglib.transform.Function.html


.. badges images and links:
.. |tests| image:: https://github.com/audeering/auglib/workflows/Test/badge.svg
    :target: https://github.com/audeering/auglib/actions?query=workflow%3ATest
    :alt: Test status
.. |coverage| image:: https://codecov.io/gh/audeering/auglib/branch/main/graph/badge.svg?token=3J0sF7GQhA
    :target: https://codecov.io/gh/audeering/auglib/
    :alt: code coverage
.. |docs| image:: https://img.shields.io/pypi/v/auglib?label=docs
    :target: https://audeering.github.io/auglib/
    :alt: auglib's documentation
.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: https://github.com/audeering/auglib/blob/main/LICENSE
    :alt: auglib's MIT license
.. |python-versions| image:: https://img.shields.io/pypi/pyversions/auglib.svg
    :target: https://pypi.org/project/auglib/
    :alt: auglib's supported Python versions
