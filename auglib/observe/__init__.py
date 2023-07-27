r"""Observable classes.

Observable objects provide a mechanism to randomize augmentation.
Instead of holding a fixed value,
an observable object reveals its value only when its called.
E.g. in the following
``observed_value_1`` and ``observed_value_2``
are randomly assigned an integer in range ``[0, 5)``:

.. code-block:: python

    observable_value = auglib.observe.IntUni(low=0, high=5)
    observed_value_1 = observable_value()  # value in [0, 5)
    observed_value_2 = observable_value()  # value in [0, 5)


By deriving from :class:`auglib.observe.Base`
we can implement our own observable class.
For instance,
we can rewrite above example as follows:

.. code-block:: python

    class MyRandom05(auglib.observe.Base):
        def __call__(self) -> int:
            return random.randint(0, 5)


    observable_value = MyRandom05()
    observed_value_1 = observable_value()  # value in [0, 5)
    observed_value_2 = observable_value()  # value in [0, 5)


Observable objects can be used in any place,
where argument support the type :class:`auglib.observe.Base`.
For instance,
to augment files with
:class:`auglib.transform.PinkNoise`
at different intensities,
we can do:

.. code-block:: python

    noise_with_random_gain = auglib.transform.PinkNoise(
        gain_db=auglib.observe.FloatUni(-30, -10),
    )
    augment = auglib.Augment(noise_with_random_gain)

    # augment each file with a different gain in (-30, -10) db
    augment.process_files(files)



"""
from auglib.core.observe import Base
from auglib.core.observe import Bool
from auglib.core.observe import FloatNorm
from auglib.core.observe import FloatUni
from auglib.core.observe import IntUni
from auglib.core.observe import List
from auglib.core.observe import observe
