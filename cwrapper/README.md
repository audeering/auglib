# C wrapper for the auglib C++ interface

In order to build the C wrapper:

1. Clone and build `auglib` in *release* mode (follow instructions on https://gitlab.audeering.com/tools/auglib)
2. Set $AUGLIB to the path where the `auglib` repo was cloned:
    ```
    export AUGLIB=</path/to/auglib>
    ```
3. Install `patchelf`: 
    ```
    sudo apt-get install -y patchelf
    ```
4. Run the build script contained in this folder:
    ```
    bash build.sh
    ```