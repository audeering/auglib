#include <auglib.h>
#include <mutex>  


#define MAX_MSG_LEN 2048

#define CALL(expr)                      \
    try                                 \
    {                                   \
        (expr);                         \
    }                                   \
    catch (const std::exception& ex)    \
    {                                   \
        _set_exception(ex);             \
    }


extern "C"
{    
    std::mt19937 auglib_randEngine(1);
    

    bool _has_exception = false;
    char _exception_msg[MAX_MSG_LEN];
    std::mutex _exception_mutex; 

    void _copy_string(char *dst, size_t size, const char *src)
    {
        size_t n = strlen(src);
        if (n >= size)
        {
            n = size - 1;
        }            
        memcpy(dst, src, n); 
        dst[n] = '\0';
    }
    
    void _set_exception(const std::exception &ex)
    {
        std::lock_guard<std::mutex> lock(_exception_mutex);
        // first come, first stored...        
        if (!_has_exception)
        {         
            _has_exception = true;
            _copy_string(_exception_msg, MAX_MSG_LEN, ex.what());          
        }
    }

    bool auglib_check_exception(char *buffer, size_t size)
    {
        std::lock_guard<std::mutex> lock(_exception_mutex);
        if (_has_exception)
        {
            _copy_string(buffer, size, _exception_msg);
        }
        else
        {
            buffer[0] = '\0';
        }        
        return _has_exception;
    }

    void auglib_release_exception()
    {
        std::lock_guard<std::mutex> lock(_exception_mutex);        
        _has_exception = false;
    }

    void auglib_random_seed(int seed = 0)
    {
        if (seed == 0)
        {
            seed = time(NULL);            
        }
        
        auglib_randEngine.seed(seed);
        srand(seed);
    }
    
    cAudioBuffer *AudioBuffer_new(size_t length_samples, unsigned int sampling_rate)
    {
        return new cAudioBuffer(length_samples, sampling_rate);
    }

    void AudioBuffer_free(cAudioBuffer *obj)
    {
        delete obj;
    }

    const float *AudioBuffer_data(cAudioBuffer *obj) 
    {
        return obj->data();
    }

    void AudioBuffer_dump(cAudioBuffer *obj)
    {        
        const sample_t *data = obj->data();
        for (size_t i = 0; i < obj->size(); i++)
        {
            printf("%lf ", data[i]);
        }
        printf("\n");
    }

    size_t AudioBuffer_size(cAudioBuffer *obj)
    {
        return obj->size();
    }

    void AudioBuffer_mix(cAudioBuffer *obj, const cAudioBuffer *auxBuf,
      float gainBase_dB = 0.0, float gainAux_dB = 0.0, size_t writePos_base = 0,
      size_t readPos_aux = 0, size_t readLength_aux = 0, bool clipMix = false,
      bool loopAux = false, bool extendBase = false)
    {
        CALL(obj->mix(*auxBuf, gainBase_dB, gainAux_dB, writePos_base, readPos_aux, 
                      readLength_aux, clipMix, loopAux, extendBase))        
    }

    void AudioBuffer_append(cAudioBuffer *obj, cAudioBuffer *aux, size_t readPos_aux = 0, size_t readLength_aux = 0)
    {
        CALL(obj->append(*aux, readPos_aux, readLength_aux))
    }

    void AudioBuffer_trim(cAudioBuffer *obj, size_t start, size_t length = 0)
    {
        CALL(obj->trim(start, length))
    }

    void AudioBuffer_addWhiteNoiseGaussian(cAudioBuffer *obj, float gain_dB, float stddev = 0.3)
    {
        CALL(obj->addWhiteNoiseGaussian(gain_dB, auglib_randEngine, stddev, false))
    }

    void AudioBuffer_addWhiteNoiseUniform(cAudioBuffer *obj, float gain_dB)
    {
        CALL(obj->addWhiteNoiseUniform(gain_dB, auglib_randEngine, false))
    }

    void AudioBuffer_addPinkNoise(cAudioBuffer *obj, float gain_dB)
    {        
        CALL(obj->addPinkNoise(gain_dB, auglib_randEngine, false))
    }

    void AudioBuffer_addTone(cAudioBuffer *obj, float freq, float gain_dB, int shape = 0, float freqLFO = 0, float LFOrange = 0)
    {
        std::string shape_s = "sine";
        switch (shape)
        {
            case 1:
                shape_s = "square";
                break;
            case 2:
                shape_s = "triangle";
                break;
            case 3:
                shape_s = "sawtooth";
                break;
        }
         CALL(obj->addTone(freq, gain_dB, shape_s, freqLFO, LFOrange, false))
    }

    void AudioBuffer_fftConvolve(cAudioBuffer *obj, cAudioBuffer *aux, bool keepTail = true)
    {
         CALL(obj->fftConvolve(*aux, keepTail))
    }

    void AudioBuffer_butterworthLowPassFilter(cAudioBuffer *obj, float cutFreq, int order = 1)
    {
        CALL(obj->butterworthLowPassFilter(cutFreq, order))
    }

    void AudioBuffer_butterworthHighPassFilter(cAudioBuffer *obj, float cutFreq, int order = 1)
    {
        CALL(obj->butterworthHighPassFilter(cutFreq, order))
    }

    void AudioBuffer_butterworthBandPassFilter(cAudioBuffer *obj, float centerFreq, float bandwidth, int order = 1)
    {
        CALL(obj->butterworthBandPassFilter(centerFreq, bandwidth, order))
    }

    void AudioBuffer_butterworthBandStopFilter(cAudioBuffer *obj, float centerFreq, float bandwidth, int order = 1)
    {
        CALL(obj->butterworthBandStopFilter(centerFreq, bandwidth, order))
    }

    void AudioBuffer_clip(cAudioBuffer *obj, float threshold = 0.0, bool soft = false, bool normalize = false)
    {
        CALL(obj->clip(threshold, soft, normalize))
    }

    void AudioBuffer_clipByRatio(cAudioBuffer *obj, float ratio, bool soft = false, bool normalize = false)
    {
        CALL(obj->clipByRatio(ratio, soft, normalize))
    }

    void AudioBuffer_normalizeByPeak(cAudioBuffer *obj, float peak_db = 0.0, bool clip = false)
    {
        CALL(obj->normalizeByPeak(peak_db, clip))
    }

    void AudioBuffer_gainStage(cAudioBuffer *obj, float gain_dB, bool clip = false)
    {
        CALL(obj->gainStage(gain_dB, clip))
    }

    void AudioBuffer_gainStageSafe(cAudioBuffer *obj, float gain_dB, float maxPeak_dB = 0.0)
    {
        CALL(obj->gainStageSafe(gain_dB, maxPeak_dB))
    }

    void AudioBuffer_compressDynamicRange(cAudioBuffer *obj, float threshold_dB, float ratio, float tAtt_s = 0.01, float tRel_s = 0.02, float kneeRadius_dB = 4.0, float makeup_dB = 0.0, bool clip = false)
    {
        CALL(obj->compressDynamicRange(threshold_dB, ratio, tAtt_s, tRel_s, kneeRadius_dB, makeup_dB, clip))
    }

    float AudioBuffer_getPeak(cAudioBuffer *obj)
    {
        return obj->getPeak();
    }

    float AudioBuffer_getPeakDecibels(cAudioBuffer *obj)
    {
        return obj->getPeakDecibels();
    }

    void AudioBuffer_AMRNB(cAudioBuffer *obj, int bitRate, int dtx = 0)
    {
        CALL(obj->AMRNB(bitRate, dtx))
    }
}
