# C wrapper for the auglib C++ interface

In order to build the C wrapper:

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