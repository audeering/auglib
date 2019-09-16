Contributing
============

If you would like to add new functionality fell free to create a `merge
request`_ . If you find errors, omissions, inconsistencies or other things
that need improvement, please create an issue_.
Contributions are always welcome!

.. _issue:
    http://gitlab2.audeering.local/jwagner/pyauglib/issues/new?issue%5BD=
.. _merge request:
    http://gitlab2.audeering.local/jwagner/pyauglib/merge_requests/new

Development Installation
------------------------

Instead of pip-installing the latest release from PyPI, you should get the
newest development version from Gitlab_::

    git clone git@srv-app-01.audeering.local:jwagner/pyauglib.git
    cd pyauglib
    # Use virtual environment
    pip install -r requirements.txt

.. _Gitlab: http://gitlab2.audeering.local/jwagner/pyauglib

This way, your installation always stays up-to-date, even if you pull new
changes from the Gitlab repository.

Building the Documentation
--------------------------

If you make changes to the documentation, you can re-create the HTML pages
using Sphinx_.
You can install it and a few other necessary packages with::

    pip install -r requirements.txt
    pip install -r docs/requirements.txt

To create the HTML pages, use::

    python -m sphinx docs/ build/sphinx/html -b html

The generated files will be available in the directory ``build/sphinx/html/``.

It is also possible to automatically check if all links are still valid::

    python -m sphinx docs/ build/sphinx/linkcheck -b linkcheck

.. _Sphinx: http://sphinx-doc.org/

Running the Tests
-----------------

You'll need pytest_ for that.
It can be installed with::

    pip install -r tests/requirements.txt

To execute the tests, simply run::

    pytest tests/

.. _pytest: https://pytest.org/

Creating a New Release
----------------------

New releases are made using the following steps:

#. Update ``CHANGELOG.rst``
#. Commit those changes as "Release X.Y.Z"
#. Create an (annotated) tag with ``git tag -a vX.Y.Z``
#. Clear the ``dist/`` directory
#. Create a distribution package with ``python setup.py sdist bdist_wheel``
#. Check that both files have the correct content
#. Upload them to PyPI_ with twine_:
   ``python -m twine upload --repository local dist/*``
#. Push the commit and the tag to Gitlab

.. _PyPI: http://artifactory.audeering.local/artifactory/api/pypi/pypi-local/simple/
.. _twine: https://twine.readthedocs.io/
