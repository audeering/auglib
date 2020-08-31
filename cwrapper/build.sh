rm ./cauglib.o
rm ../auglib/bin/*.so

g++ -c -fPIC cauglib.c -o cauglib.o \
-I$AUGLIB \
-I$AUGLIB/soundtouch/include \
-I$AUGLIB/FFTConvolver \
-I$AUGLIB/DSPFilters/include

g++ -shared -Wl,-zdefs -o ../auglib/core/bin/libcauglib.so cauglib.o \
-L$AUGLIB/Release \
-L$AUGLIB/soundtouch/source/SoundTouch/.libs \
-L$AUGLIB/Release/FFTConvolver \
-L$AUGLIB/Release/DSPFilters \
-laugdeering -lopencore-amrnb -lFFTConvolver -lSoundTouch -lDSPFilters \
-Wl,-rpath=$AUGLIB/soundtouch/source/SoundTouch/.libs

function patch_strings_in_file() {
    local FILE="$1"
    local PATTERN="$2"
    local REPLACEMENT="$3"

    # Find all unique strings in FILE that contain the pattern
    STRINGS=$(strings ${FILE} | grep ${PATTERN} | sort -u -r)

    if [ "${STRINGS}" != "" ] ; then
        echo "File '${FILE}' contain strings with '${PATTERN}' in them:"

        for OLD_STRING in ${STRINGS} ; do
            # Create the new string with a simple bash-replacement
            NEW_STRING=${OLD_STRING//${PATTERN}/${REPLACEMENT}}

            # Create null terminated ASCII HEX representations of the strings
            OLD_STRING_HEX="$(echo -n ${OLD_STRING} | xxd -g 0 -u -ps -c 256)00"
            NEW_STRING_HEX="$(echo -n ${NEW_STRING} | xxd -g 0 -u -ps -c 256)00"

            if [ ${#NEW_STRING_HEX} -le ${#OLD_STRING_HEX} ] ; then
                # Pad the replacement string with null terminations so the
                # length matches the original string
                while [ ${#NEW_STRING_HEX} -lt ${#OLD_STRING_HEX} ] ; do
                    NEW_STRING_HEX="${NEW_STRING_HEX}00"
                done

                # Now, replace every occurrence of OLD_STRING with NEW_STRING
                echo -n "Replacing ${OLD_STRING} with ${NEW_STRING}... "
                hexdump -ve '1/1 "%.2X"' ${FILE} | \
                sed "s/${OLD_STRING_HEX}/${NEW_STRING_HEX}/g" | \
                xxd -r -p > ${FILE}.tmp
                chmod --reference ${FILE} ${FILE}.tmp
                mv ${FILE}.tmp ${FILE}
                echo "Done!"
            else
                echo "New string '${NEW_STRING}' is longer than old" \
                     "string '${OLD_STRING}'. Skipping."
            fi
        done
    fi
}

# remove absolute path to soundtouch library
patch_strings_in_file ../auglib/core/bin/libcauglib.so "$AUGLIB/soundtouch/source/SoundTouch/\.libs" ""

# add bin folder to r-path
# https://nixos.org/patchelf.html
# https://stackoverflow.com/questions/39978762/linux-executable-cant-find-shared-library-in-same-folder
patchelf --set-rpath '${ORIGIN}' ../auglib/core/bin/libcauglib.so
