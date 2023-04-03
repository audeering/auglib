import numpy as np
import pytest

import audmath
import audobject
import auglib


@pytest.mark.parametrize(
    'base_dur,aux_dur,sr,unit',
    [
        (1.0, 1.0, 8000, None),
        (16000, 8000, 16000, 'samples'),
        (500, 1000, 44100, 'ms'),
    ],
)
def test_Mix_1(tmpdir, base_dur, aux_dur, sr, unit):

    unit = unit or 'seconds'
    n_base = auglib.utils.to_samples(base_dur, sampling_rate=sr, unit=unit)
    n_aux = auglib.utils.to_samples(aux_dur, sampling_rate=sr, unit=unit)

    n_min = min(n_base, n_aux)

    # init auxiliary buffer
    with auglib.AudioBuffer(aux_dur, sr, value=1.0, unit=unit) as aux:

        # default mix
        transform = auglib.transform.Mix(aux)
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with auglib.AudioBuffer(base_dur, sr, unit=unit) as base:
            transform(base)
            expected_mix = np.concatenate(
                [np.ones(n_min), np.zeros(n_base - n_min)]
            )
            np.testing.assert_equal(base._data, expected_mix)

        # clipping
        transform = auglib.transform.Mix(
            aux,
            gain_aux_db=audmath.db(2),
            loop_aux=True,
            clip_mix=True,
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with auglib.AudioBuffer(base_dur, sr, unit=unit) as base:
            transform(base)
            expected_mix = np.ones(n_base)
            np.testing.assert_equal(base._data, expected_mix)

    # Test for repeated execution.
    values = [0, 1, 2, 3, 4]
    expected_mix = np.zeros(n_base)
    for n in range(len(values)):
        expected_mix += np.concatenate(
            [values[n:], np.zeros(n_base - len(values[n:]))]
        )

    # Shift aux by increasing read_pos_aux
    with auglib.AudioBuffer.from_array(values, sr) as aux:
        transform = auglib.transform.Mix(
            aux,
            read_pos_aux=auglib.observe.List(values),
            unit='samples',
            num_repeat=len(values),
        )
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'auglib.core.buffer.AudioBuffer'>. "
            "As a workaround, save buffer to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            transform.to_yaml_s(include_version=False)

        with auglib.AudioBuffer(base_dur, sr, unit=unit) as base:
            transform(base)
            np.testing.assert_equal(base._data, expected_mix)

    # Shift aux by observe list of buffers
    transform = auglib.transform.Mix(
        auglib.observe.List(
            [
                auglib.AudioBuffer.from_array(values[n:], sr)
                for n in range(len(values))
            ]
        ),
        num_repeat=len(values),
    )
    error_msg = (
        "Cannot serialize list if it contains an instance "
        "of <class 'auglib.core.buffer.AudioBuffer'>. "
        "As a workaround, save buffer to disk and add filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    with auglib.AudioBuffer(base_dur, sr, unit=unit) as base:
        transform(base)
        np.testing.assert_equal(base._data, expected_mix)


# All duration are given in samples for this test
@pytest.mark.parametrize('base_duration', [5, 10])
@pytest.mark.parametrize('aux_duration', [5, 10])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('write_pos_base', [0, 1])
@pytest.mark.parametrize('extend_base', [False, True])
@pytest.mark.parametrize('read_pos_aux', [0, 1])
@pytest.mark.parametrize('read_dur_aux', [None, 3, 6])
@pytest.mark.parametrize('loop_aux', [False, True])
@pytest.mark.parametrize('gain_base_db', [0, 10])
@pytest.mark.parametrize(
    'gain_aux_db, snr_db',
    [
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ]
)
def test_Mix_2(
        base_duration,
        aux_duration,
        sampling_rate,
        write_pos_base,
        extend_base,
        read_pos_aux,
        read_dur_aux,
        loop_aux,
        gain_base_db,
        gain_aux_db,
        snr_db,
):

    aux_values = np.array(range(aux_duration))
    base_value = 0.1

    # Skip tests for loop_aux=True and read_dur_aux not None
    # as this is broken at the moment, see
    # https://gitlab.audeering.com/tools/pyauglib/-/issues/47
    if (
            loop_aux
            and read_dur_aux is not None
    ):
        return

    # Skip test for read_dur_aux longer than len(aux)
    # and gain_base_db different from 0
    # as this is borken at the moment, see
    # https://gitlab.audeering.com/tools/pyauglib/-/issues/76
    if (
            gain_base_db != 0
            and read_dur_aux is not None
            and read_dur_aux > len(aux_values)
    ):
        return

    with auglib.AudioBuffer.from_array(aux_values, sampling_rate) as aux:
        with auglib.AudioBuffer(
                base_duration,
                sampling_rate,
                value=base_value,
                unit='samples',
        ) as base:

            # Number of samples read for mix from aux
            if (
                    read_dur_aux is None
                    or read_dur_aux == 0
            ):
                len_mix_aux = len(aux) - read_pos_aux
            else:
                len_mix_aux = read_dur_aux

            # Number of samples available for mix in base
            len_mix_base = len(base) - write_pos_base

            # If number of samples available for mix in base
            # is smaller than the number of samples read from aux
            # we pad zeros to base if extend_base is `True`.
            # Otherwise we trim the aux signal.
            gain_base = auglib.utils.from_db(gain_base_db)
            expected_mix = gain_base * base_value * np.ones(len(base))
            if len_mix_aux > len_mix_base:
                if extend_base:
                    expected_mix = np.concatenate(
                        [expected_mix, np.zeros(len_mix_aux - len_mix_base)],
                    )
                else:
                    len_mix_aux = len_mix_base

            # read_dur_aux is allowed to extend aux buffer,
            # in this case zeros are padded at the end.
            # Those zeros will NOT be included in the RMS_dB calculation
            len_mix_aux = min(
                len(aux) - read_pos_aux,
                len_mix_aux,
            )
            aux_values = aux_values[read_pos_aux:read_pos_aux + len_mix_aux]

            # As we use a fixed signal for `base`
            # the RMS_db value is independent of signal length
            rms_db_base = audmath.db(audmath.rms(gain_base * base_value))
            rms_db_aux = audmath.db(audmath.rms(aux_values))

            # Get gain factor for aux
            if gain_aux_db is None:
                gain_aux_db = 0.0
            if snr_db is not None:
                gain_aux_db = rms_db_base - snr_db - rms_db_aux
            gain_aux = auglib.utils.from_db(gain_aux_db)

            # Add aux values to expected mix
            mix_start = write_pos_base
            expected_mix[mix_start:mix_start + len_mix_aux] += (
                gain_aux * aux_values
            )
            # If aux should be looped,
            # we have to repeat adding aux to the mix
            mix_start += len_mix_aux
            if loop_aux:
                while mix_start < len(expected_mix):
                    mix_end = min(
                        mix_start + len_mix_aux,
                        len(expected_mix),
                    )
                    expected_mix[mix_start:mix_end] += (
                        gain_aux * aux_values[:mix_end - mix_start]
                    )
                    mix_start += len_mix_aux

            transform = auglib.transform.Mix(
                aux,
                gain_base_db=gain_base_db,
                gain_aux_db=gain_aux_db,
                snr_db=snr_db,
                write_pos_base=write_pos_base,
                read_pos_aux=read_pos_aux,
                read_dur_aux=read_dur_aux,
                extend_base=extend_base,
                unit='samples',
                loop_aux=loop_aux,
            )
            transform(base)

            np.testing.assert_almost_equal(
                base._data,
                expected_mix,
                decimal=5,
            )
