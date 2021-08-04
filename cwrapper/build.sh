rm -rf Release
mkdir Release
cd Release
conan install ..
cmake .. -DCMAKE_BUILD_TYPE=Release -DAUGLIB_DIR=$AUGLIB
make

cd ..

cp Release/lib/libcauglib.so ../auglib/core/bin/libcauglib.so

# add bin folder to r-path
# https://nixos.org/patchelf.html
# https://stackoverflow.com/questions/39978762/linux-executable-cant-find-shared-library-in-same-folder
patchelf --set-rpath '${ORIGIN}' ../auglib/core/bin/libcauglib.so
