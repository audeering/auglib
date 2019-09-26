#include <auglib.h>


extern "C"
{
    std::mt19937 auglib_randEngine(1);

    void auglib_random_seed(int seed = 0)
    {
        if (seed == 0)
        {
            seed = time(NULL);            
        }
        
        auglib_randEngine.seed(seed);
        srand(seed);
    }
    
    cAudioBuffer<float> *AudioBuffer_new(size_t length_samples, unsigned int sampling_rate)
    {
        return new cAudioBuffer<float>(length_samples, sampling_rate);
    }

    void AudioBuffer_free(cAudioBuffer<float> *obj)
    {
        delete obj;
    }

    const float *AudioBuffer_data(cAudioBuffer<float> *obj) 
    {
        return obj->data();
    }

    size_t AudioBuffer_size(cAudioBuffer<float> *obj)
    {
        return obj->size();
    }

    void AudioBuffer_mix(cAudioBuffer<float> *obj, const cAudioBuffer<float> *auxBuf,
      float gainBase_dB = 0.0, float gainAux_dB = 0.0, size_t writePos_base = 0,
      size_t readPos_aux = 0, size_t readLength_aux = 0, bool clipMix = false,
      bool loopAux = false, bool extendBase = false)
    {
        obj->mix(*auxBuf, gainBase_dB, gainAux_dB, writePos_base, readPos_aux,
                 readLength_aux, clipMix, loopAux, extendBase);
    }

    void AudioBuffer_append(cAudioBuffer<float> *obj, cAudioBuffer<float> *aux, size_t readPos_aux = 0, size_t readLength_aux = 0)
    {
        obj->append(*aux, readPos_aux, readLength_aux);
    }

    void AudioBuffer_addWhiteNoiseGaussian(cAudioBuffer<float> *obj, float gain_dB, float stddev = 0.3)
    {
        obj->addWhiteNoiseGaussian(gain_dB, auglib_randEngine, stddev, false);
    }


    void AudioBuffer_addWhiteNoiseUniform(cAudioBuffer<float> *obj, float gain_dB)
    {
        obj->addWhiteNoiseUniform(gain_dB, auglib_randEngine, false);
    }

    void AudioBuffer_addPinkNoise(cAudioBuffer<float> *obj, float gain_dB)
    {        
        obj->addPinkNoise(gain_dB, auglib_randEngine, false);
    }

    void AudioBuffer_addTone(cAudioBuffer<float> *obj, float freq, float gain_dB, int shape = 0, float freqLFO = 0, float LFOrange = 0)
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
        obj->addTone(freq, gain_dB, shape_s, freqLFO, LFOrange, false);
    }

    void AudioBuffer_fftConvolve(cAudioBuffer<float> *obj, cAudioBuffer<float> *aux, bool keepTail = true)
    {
        obj->fftConvolve(*aux, keepTail);
    }

    void AudioBuffer_butterworthLowPassFilter(cAudioBuffer<float> *obj, float cutFreq, int order = 1)
    {
        obj->butterworthLowPassFilter(cutFreq, order);
    }

    void AudioBuffer_butterworthHighPassFilter(cAudioBuffer<float> *obj, float cutFreq, int order = 1)
    {
        obj->butterworthHighPassFilter(cutFreq, order);
    }

    void AudioBuffer_butterworthBandPassFilter(cAudioBuffer<float> *obj, float centerFreq, float bandwidth, int order = 1)
    {
        obj->butterworthBandPassFilter(centerFreq, bandwidth, order);
    }

    void AudioBuffer_clip(cAudioBuffer<float> *obj, float threshold = 0.0, bool soft = false, bool normalize = false)
    {
        obj->clip(threshold, soft, normalize);
    }

    void AudioBuffer_clipByRatio(cAudioBuffer<float> *obj, float ratio, bool soft = false, bool normalize = false)
    {
        obj->clipByRatio(ratio, soft, normalize);
    }

    void AudioBuffer_normalizeByPeak(cAudioBuffer<float> *obj, float peak_db = 0.0, bool clip = false)
    {
        obj->normalizeByPeak(peak_db, clip);
    }

    void AudioBuffer_gainStage(cAudioBuffer<float> *obj, float gain_dB, bool clip = false)
    {
        obj->gainStage(gain_dB, clip);
    }
        
}