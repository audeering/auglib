======
auglib
======

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


.. _datasets: https://audeering.github.io/usage.html#augment-a-database-to-disk
.. _examples: https://audeering.github.io/examples.html
.. _external: https://audeering.github.io/external.html
.. _files: https://audeering.github.io/usage.html#augment-files-in-memory
.. _installation: https://audeering.github.io/install.html
.. _signals: https://audeering.github.io/usage.html#augment-a-signal
.. _transforms: https://audeering.github.io/api/auglib.transform.html
.. _usage: https://audeering.github.io/usage.html
.. _user defined: https://audeering.github.io/api/auglib.transform.Function.html
