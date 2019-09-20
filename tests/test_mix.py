import numpy as np
import pytest

from auglib import AudioBuffer
from auglib.utils import gain_to_db, dur_to_samples


@pytest.mark.parametrize('base_dur,aux_dur,sr,unit',
                         [(1.0, 1.0, 8000, None),
                          (16000, 8000, 16000, 'samples'),
                          (500, 1000, 44100, 'ms')])
def test_mix(base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = dur_to_samples(base_dur, sr, unit=unit)
    n_aux = dur_to_samples(aux_dur, sr, unit=unit)

    n_min = min(n_base, n_aux)
    n_max = max(n_base, n_aux)

    # init auxiliary buffer

    aux = AudioBuffer(aux_dur, sr, unit=unit)
    aux.data += 1

    # default mix

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux)
        assert base.data[n_min:].sum() == 0
        np.testing.assert_equal(base.data[:n_min], aux.data[:n_min])

    # loop auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, loop_aux=True)
        np.testing.assert_equal(base.data, np.ones(n_base))

    # extend base

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, extend_base=True)
        assert len(base) == n_max
        np.testing.assert_equal(base.data[:n_aux], np.ones(n_aux))

    # restrict length of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, read_dur_aux=1, unit='samples')
        assert base.data[0] == 1 and base.data.sum() == 1

    # read position of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, read_pos_aux=n_aux - 1, unit='samples')
        assert base.data[0] == 1 and base.data.sum() == 1

    # write position of base

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, write_pos_base=n_base - 1, unit='samples')
        assert base.data[-1] == 1 and base.data.sum() == 1

    # set gain of auxiliary

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, gain_aux_db=gain_to_db(2), loop_aux=True)
        assert all(base.data == 2)
        base.mix(aux, gain_base_db=gain_to_db(0.5), loop_aux=True)
        assert all(base.data == 2)

    # clipping

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.mix(aux, gain_aux_db=gain_to_db(2), loop_aux=True, clip_mix=True)
        assert all(base.data == 1)

    aux.free()


@pytest.mark.parametrize('base_dur,aux_dur,sr,unit',
                         [(1.0, 1.0, 8000, None),
                          (16000, 8000, 16000, 'samples'),
                          (500, 1000, 44100, 'ms')])
def test_append(base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = dur_to_samples(base_dur, sr, unit=unit)
    n_aux = dur_to_samples(aux_dur, sr, unit=unit)

    aux = AudioBuffer(aux_dur, sr, unit=unit)
    aux.data += 1

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.append(aux)
        np.testing.assert_equal(base.data[:n_base], np.zeros(n_base))
        np.testing.assert_equal(base.data[n_base:], np.ones(n_aux))

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.append(aux, read_pos_aux=n_aux - 1)
        np.testing.assert_equal(base.data[:n_base], np.zeros(n_base))
        assert len(base.data) == n_base + 1
        assert base.data[-1] == 1

    with AudioBuffer(base_dur, sr, unit=unit) as base:
        base.append(aux, read_dur_aux=1, unit='samples')
        np.testing.assert_equal(base.data[:n_base], np.zeros(n_base))
        assert len(base.data) == n_base + 1
        assert base.data[-1] == 1

    aux.free()
