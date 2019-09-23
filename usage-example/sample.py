import audiofile as af
from auglib import AudioBuffer, random_seed, PinkNoise


# Seed the random engine
random_seed(1)

with AudioBuffer.from_file('audio/11a04Ac.wav') as base:

    # append 5s of silence
    with AudioBuffer(5.0, base.sampling_rate) as aux:
        base.append(aux)

    # append an additional utterance (amplified by 4dB)
    with AudioBuffer.from_file('audio/11a01Wc.wav') as aux:
        aux.gain_stage(4.0)
        base.append(aux)

    # append some more silence
    with AudioBuffer(5.0, base.sampling_rate) as aux:
        base.append(aux)

    # add some background noise (open air swimming area)
    with AudioBuffer.from_file('audio/babblenoise.wav') as aux:
        # normalise to peak amplitude
        aux.normalize_by_peak()
        # boosting by 2dB, reading from 20s into the file
        base.mix(aux, gain_aux_db=2.0, read_pos_aux=20.0)

    # add some music, simulating a small speaker
    with AudioBuffer.from_file('audio/music.wav') as aux, \
            AudioBuffer.from_file('ir/small-speaker_16kHz.wav') as ir:
        aux.fft_convolve(ir, keep_tail=False)
        base.mix(aux, gain_aux_db=-12.0, write_pos_base=3.0, loop_aux=True)

    # add reverb to simulate a large hall
    with AudioBuffer.from_file('ir/factory-hall_16kHz.wav') as ir:
        base.fft_convolve(ir)

    # cheap mic simulation (bandpass-filter + pink noise)
    base.band_pass(1, 1000.0, 1800.0)
    with PinkNoise(len(base), base.sampling_rate, unit='samples') as noise:
        base.mix(noise, gain_aux_db=-30.0)

    # clip and save
    base.clip()
    base.to_file('sample.wav')
