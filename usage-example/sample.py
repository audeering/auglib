from auglib import AudioBuffer
from auglib.utils import random_seed
from auglib.transform import Append, AppendValue, GainStage, NormalizeByPeak, \
    FFTConvolve, BandPass, Mix, Clip, PinkNoise, Compose


def main():

    # Seed the random engine
    random_seed(1)

    transform = Compose([
        AppendValue(5.0),
        Append('speech/11a01Wc.wav',
               transform=GainStage(4.0)),
        AppendValue(5.0),
        Mix('noise/babble.wav',
            gain_aux_db=2.0,
            read_pos_aux=20.0,
            transform=NormalizeByPeak()),
        Mix('noise/music.wav',
            gain_aux_db=-12.0,
            write_pos_base=3.0,
            loop_aux=True,
            transform=FFTConvolve('ir/small-speaker.wav',
                                  keep_tail=True)),
        FFTConvolve('ir/factory-hall.wav'),
        BandPass(1000.0, 1800.0),
        PinkNoise(gain_db=-30.0),
        Clip(),
    ])

    transform.dump('sample')
    transform = Compose.load('sample')

    with AudioBuffer.read('speech/11a04Ac.wav') as buf:
        transform(buf)
        buf.write('sample.wav')


if __name__ == '__main__':

    main()
