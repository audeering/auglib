cmake_minimum_required(VERSION 3.21.1)

project(cAuglib)

# setup conan
if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(KEEP_RPATHS)
endif()

set(AUGLIB_DIR "" CACHE PATH "Path to the auglib directory.")

add_library(cauglib SHARED cauglib.cpp)

target_include_directories(cauglib PRIVATE 
    ${CONAN_INCLUDE_DIRS}
    ${AUGLIB_DIR}/soundtouch/include
    ${AUGLIB_DIR}/FFTConvolver
    ${AUGLIB_DIR}
)

find_library(SoundTouch SoundTouch
    NAMES libSoundTouch
    HINTS "${AUGLIB_DIR}/soundtouch/source/SoundTouch/.libs"
    REQUIRED
)

find_library(FFTConvolver FFTConvolver
    NAMES libFFTConvolver
    HINTS "${AUGLIB_DIR}/Release/lib"
    REQUIRED
)

find_library(opencore-amrnb opencore-amrnb REQUIRED)

find_library(augdeering augdeering
    NAMES libaugdeering
    HINTS "${AUGLIB_DIR}/Release/lib"
    REQUIRED
)

# Note: the order of the libraries matters!
target_link_libraries(cauglib PRIVATE
    ${augdeering}  # This needs to be the first in the list!
    ${CONAN_LIBS}
    ${FFTConvolver}
    ${opencore-amrnb}
    ${SoundTouch}
)
