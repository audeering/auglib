Contributing
============

If you would like to add new functionality fell free to create a `merge
request`_ . If you find errors, omissions, inconsistencies or other things
that need improvement, please create an issue_.
Contributions are always welcome!

.. _issue:
    https://gitlab.audeering.com/tools/pyauglib/issues/new?issue%5BD=
.. _merge request:
    https://gitlab.audeering.com/tools/pyauglib/merge_requests/new


Development Installation
------------------------

Instead of pip-installing the latest release from PyPI, you should get the
newest development version from Gitlab_::

    git clone git@srv-app-01.audeering.local:tools/pyauglib.git
    cd pyauglib
    # Use virtual environment
    pip install -r requirements.txt

.. _Gitlab: https://gitlab.audeering.com/tools/pyauglib

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

.. _Sphinx: https://sphinx-doc.org/


Running the Tests
-----------------

You'll need pytest_ for that.
It can be installed with::

    pip install -r tests/requirements.txt

To execute the tests, simply run::

    pytest tests/

.. _pytest: https://pytest.org/


Updating the C++ Library
------------------------

``auglib`` depends on the
`C++ auglib library`_,
which is included in ``auglib``
with the help of a C-wrapper.

To use a new version of the
`C++ auglib library`_,
you need to update the C-wrapper.
Install needed requirements by::

    pip install -r cwrapper/requirements.txt
    sudo apt-get install --yes cmake libtool automake patchelf

Clone and install AMR codec::

    cd cwrapper
    git clone --depth 1 --branch master https://gitlab.audeering.com/tools/opencore-amr.git
    cd opencore-amr
    autoreconf --install
    autoconf
    ./configure
    make
    sudo make install
    cd ..

Setup Conan_ to find packages on Artifactory_
by following the `Setup Conan`_ instructions.

Clone and build a static version of the `C++ auglib library`_
in release mode::

    git clone --depth 1 --branch main https://gitlab.audeering.com/tools/auglib
    cd auglib
    bash build-soundtouch.sh
    mkdir build
    cd build
    conan install ..
    cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENCOREAMR_SUPPORT=ON -DOPENCOREAMRNB_INCLUDE_PATH=/usr/local/include/opencore-amrnb
    cd ../..

And finally build the C-wrapper::

    export AUGLIB=./auglib
    bash build.sh

.. _C++ auglib library: https://gitlab.audeering.com/tools/auglib
.. _Conan: https://conan.io
.. _Artifactory: https://artifactory.audeering.com/ui/repos/tree/General/conan-local/
.. _Setup Conan: https://gitlab.audeering.com/devops/conan/meta/-/blob/master/conan-setup.md


Including a New C++ Function
----------------------------

If you added a new function in the
`C++ auglib library`_
and would like to include it in ``auglib``
as well,
you need to include it in ``cwrapper/cauglib.cpp``
and follow the steps
explained in the previous section
on updating the C++ library.

Afterwards,
add the functions signature to ``auglib/core/api.py``
as well.


Creating a New Release
----------------------

New releases are made using the following steps:

#. Update ``CHANGELOG.rst``
#. Commit those changes as "Release X.Y.Z"
#. Create an (annotated) tag with ``git tag -a vX.Y.Z``
#. Make sure you have an `artifactory-tokenizer`_ project
#. Push the commit and the tag to Gitlab

.. _artifactory-tokenizer:
    https://gitlab.audeering.com/devops/artifactory/tree/master/token
