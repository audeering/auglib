# C wrapper for auglib's C++ interface

## Conan-related preparation

The C wrapper is intended to be built via CMake, also using Conan as a 
dependency manager (although not for all dependencies). 
For the Conan part, it is suggested to set up and activate a Python virtual 
environment:

```bash
virtualenv --python=python3 venv
source venv/bin/activate
```

Once the virtual environment is activated, install the `conan` package:

```bash
pip install conan
```

If you have never used Conan in audEERING, chances are you will need to set up 
Artifactory as remote. Please consult the internal documentation about Conan:
https://gitlab.audeering.com/devops/conan/meta.

## Actual steps for building the C wrapper

Once you have Conan set up correctly, in order to build the C wrapper:

1. Clone, build, and install `opencore-amr`:
    * `apt-get install --yes cmake`
    * `apt-get install --yes libtool`
    * `apt-get install --yes automake`
    * `git clone https://git.code.sf.net/p/opencore-amr/code opencore-amr-code`
    * `cd opencore-amr-code`
    * `autoreconf --install`
    * `autoconf`
    * `./configure && make && sudo make install`
2. Clone and build `auglib` (static version) in *release* mode (follow
 instructions on https://gitlab.audeering.com/tools/auglib)
3. Set $AUGLIB to the path where the `auglib` repo was cloned:
    ```
    export AUGLIB=</path/to/auglib>
    ```
4. Install `patchelf`: 
    ```
    sudo apt-get install -y patchelf
    ```
5. Run the build script contained in this folder:
    ```
    bash build.sh
    ```