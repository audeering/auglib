import os
import ctypes


lib_root = os.path.dirname(os.path.realpath(__file__))
if os.name == 'nt':
    lib_path = os.path.join(
        lib_root, 'bin', 'cauglib.dll',
    )  # pragma: no cover
else:
    lib_path = os.path.join(
        lib_root, 'bin', 'libcauglib.so',
    )  # pragma: no cover
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.AudioBuffer_new.argtypes = [ctypes.c_uint, ctypes.c_size_t]
lib.AudioBuffer_new.restype = ctypes.c_void_p

lib.AudioBuffer_free.argtypes = [ctypes.c_void_p]
lib.AudioBuffer_free.restype = ctypes.c_void_p

lib.AudioBuffer_dump.argtypes = [ctypes.c_void_p]
lib.AudioBuffer_dump.restype = ctypes.c_void_p

lib.AudioBuffer_data.argtypes = [ctypes.c_void_p]
lib.AudioBuffer_data.restype = ctypes.POINTER(ctypes.c_float)

lib.AudioBuffer_size.argtypes = [ctypes.c_void_p]
lib.AudioBuffer_size.restype = ctypes.c_size_t

lib.auglib_check_exception.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
lib.auglib_check_exception.restype = ctypes.c_bool

lib.auglib_release_exception.argtypes = []
lib.auglib_release_exception.restype = ctypes.c_void_p

lib.AudioBuffer_mix.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_float, ctypes.c_float,
                                ctypes.c_size_t, ctypes.c_size_t,
                                ctypes.c_size_t, ctypes.c_bool,
                                ctypes.c_bool, ctypes.c_bool]
lib.AudioBuffer_mix.restypes = ctypes.c_void_p

lib.AudioBuffer_append.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_size_t]
lib.AudioBuffer_append.restype = ctypes.c_void_p

lib.auglib_random_seed.argtypes = [ctypes.c_int]
lib.auglib_random_seed.restype = ctypes.c_void_p

lib.AudioBuffer_addWhiteNoiseGaussian.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_float,
                                                  ctypes.c_float]
lib.AudioBuffer_addWhiteNoiseGaussian.restype = ctypes.c_void_p

lib.AudioBuffer_addWhiteNoiseUniform.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_float]
lib.AudioBuffer_addWhiteNoiseUniform.restype = ctypes.c_void_p

lib.AudioBuffer_addPinkNoise.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.AudioBuffer_addPinkNoise.restype = ctypes.c_void_p

lib.AudioBuffer_addTone.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                    ctypes.c_float, ctypes.c_int,
                                    ctypes.c_float, ctypes.c_float]
lib.AudioBuffer_addTone.restype = ctypes.c_void_p

lib.AudioBuffer_fftConvolve.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_bool]
lib.AudioBuffer_fftConvolve.restype = ctypes.c_void_p

lib.AudioBuffer_butterworthLowPassFilter.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_float,
                                                     ctypes.c_int]
lib.AudioBuffer_butterworthLowPassFilter.restype = ctypes.c_void_p

lib.AudioBuffer_butterworthHighPassFilter.argtypes = [ctypes.c_void_p,
                                                      ctypes.c_float,
                                                      ctypes.c_int]
lib.AudioBuffer_butterworthHighPassFilter.restype = ctypes.c_void_p

lib.AudioBuffer_butterworthBandPassFilter.argtypes = [ctypes.c_void_p,
                                                      ctypes.c_float,
                                                      ctypes.c_float,
                                                      ctypes.c_int]
lib.AudioBuffer_butterworthBandPassFilter.restype = ctypes.c_void_p

lib.AudioBuffer_butterworthBandStopFilter.argtypes = [ctypes.c_void_p,
                                                      ctypes.c_float,
                                                      ctypes.c_float,
                                                      ctypes.c_int]
lib.AudioBuffer_butterworthBandStopFilter.restype = ctypes.c_void_p

lib.AudioBuffer_clip.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                 ctypes.c_bool, ctypes.c_bool]
lib.AudioBuffer_clip.restype = ctypes.c_void_p

lib.AudioBuffer_clipByRatio.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                        ctypes.c_bool, ctypes.c_bool]
lib.AudioBuffer_clipByRatio.restype = ctypes.c_void_p

lib.AudioBuffer_normalizeByPeak.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                            ctypes.c_bool]
lib.AudioBuffer_normalizeByPeak.restype = ctypes.c_void_p

lib.AudioBuffer_gainStage.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                      ctypes.c_bool]
lib.AudioBuffer_gainStage.restype = ctypes.c_void_p

lib.AudioBuffer_gainStageSafe.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                          ctypes.c_float]
lib.AudioBuffer_gainStageSafe.restype = ctypes.c_void_p

lib.AudioBuffer_compressDynamicRange.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_float,
                                                 ctypes.c_float,
                                                 ctypes.c_float,
                                                 ctypes.c_float,
                                                 ctypes.c_float,
                                                 ctypes.c_float,
                                                 ctypes.c_bool]
lib.AudioBuffer_compressDynamicRange.restype = ctypes.c_void_p

lib.AudioBuffer_getPeak.argtypes = [ctypes.c_void_p]
lib.AudioBuffer_getPeak.restype = ctypes.c_float

lib.AudioBuffer_getPeakDecibels.argtypes = [ctypes.c_void_p]
lib.AudioBuffer_getPeakDecibels.restype = ctypes.c_float

lib.AudioBuffer_AMRNB.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.AudioBuffer_AMRNB.restype = ctypes.c_void_p
