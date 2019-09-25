# set $AUGLIB to the installation path of auglib
# https://gitlab.audeering.com/tools/auglib
# and build in Release before running the script

g++ -c -fPIC cauglib.c -o cauglib.o \
-I$AUGLIB \
-I$AUGLIB/soundtouch/include \
-I$AUGLIB/FFTConvolver \
-I$AUGLIB/DSPFilters/include

g++ -shared -Wl,-zdefs -o ../auglib/bin/libcauglib.so cauglib.o \
-L$AUGLIB/Release \
-L$AUGLIB/soundtouch/source/SoundTouch/.libs \
-L$AUGLIB/Release/FFTConvolver \
-L$AUGLIB/Release/DSPFilters \
-laugdeering -lFFTConvolver -lSoundTouch -lDSPFilters \
-Wl,-rpath=$AUGLIB/soundtouch/source/SoundTouch/.libs
