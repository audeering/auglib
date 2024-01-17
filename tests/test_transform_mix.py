import numpy as np
import pytest

import audmath

import auglib


@pytest.mark.parametrize(
    "base_dur, aux_dur, sampling_rate, unit",
    [
        (1.0, 1.0, 8000, None),
        (16000, 8000, 16000, "samples"),
        (500, 1000, 44100, "ms"),
    ],
)
def test_mix_1(tmpdir, base_dur, aux_dur, sampling_rate, unit):
    unit = unit or "seconds"
    n_base = auglib.utils.to_samples(
        base_dur,
        sampling_rate=sampling_rate,
        unit=unit,
    )
    n_aux = auglib.utils.to_samples(
        aux_dur,
        sampling_rate=sampling_rate,
        unit=unit,
    )

    n_min = min(n_base, n_aux)

    # init base and auxiliary signals
    base = np.zeros((1, n_base))
    aux = np.ones((1, n_aux))

    # default mix
    transform = auglib.transform.Mix(aux)
    error_msg = (
        "Cannot serialize an instance "
        "of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and pass filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    expected_mix = np.concatenate(
        [np.ones((1, n_min)), np.zeros((1, n_base - n_min))],
        axis=1,
    )
    expected_mix = expected_mix.astype(auglib.core.transform.DTYPE)
    np.testing.assert_array_equal(
        transform(base, sampling_rate),
        expected_mix,
        strict=True,
    )

    # clipping
    transform = auglib.transform.Mix(
        aux,
        gain_aux_db=audmath.db(2),
        loop_aux=True,
        clip_mix=True,
    )
    error_msg = (
        "Cannot serialize an instance "
        "of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and pass filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    expected_mix = np.ones(
        (1, n_base),
        dtype=auglib.core.transform.DTYPE,
    )
    np.testing.assert_array_equal(
        transform(base, sampling_rate),
        expected_mix,
        strict=True,
    )

    # Test for repeated execution.
    values = np.array([[0, 1, 2, 3, 4]])
    expected_mix = np.zeros((1, n_base))
    for n in range(values.shape[1]):
        expected_mix += np.concatenate(
            [values[:, n:], np.zeros((1, n_base - values[:, n:].shape[1]))],
            axis=1,
        )
    expected_mix = expected_mix.astype(auglib.core.transform.DTYPE)

    # Shift aux by increasing read_pos_aux
    transform = auglib.transform.Mix(
        values,
        read_pos_aux=auglib.observe.List(values[0]),
        unit="samples",
        num_repeat=values.shape[1],
    )
    error_msg = (
        "Cannot serialize an instance "
        "of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and pass filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    np.testing.assert_array_equal(
        transform(base),
        expected_mix,
        strict=True,
    )

    # Shift aux by observe list of signals
    transform = auglib.transform.Mix(
        auglib.observe.List([values[:, n:] for n in range(values.shape[1])]),
        num_repeat=values.shape[1],
    )
    error_msg = (
        "Cannot serialize list if it contains an instance "
        "of <class 'numpy.ndarray'>. "
        "As a workaround, save signal to disk and add filename."
    )
    with pytest.raises(ValueError, match=error_msg):
        transform.to_yaml_s(include_version=False)

    np.testing.assert_array_equal(
        transform(base, sampling_rate),
        expected_mix,
        strict=True,
    )


@pytest.mark.parametrize("base_duration", [5, 10])
@pytest.mark.parametrize("aux_duration", [5, 10])
@pytest.mark.parametrize("write_pos_base", [0, 1])
@pytest.mark.parametrize("extend_base", [False, True])
@pytest.mark.parametrize("read_pos_aux", [0, 1])
@pytest.mark.parametrize("read_dur_aux", [None, 3, 6])
@pytest.mark.parametrize("loop_aux", [False, True])
@pytest.mark.parametrize("gain_base_db", [0, 10])
@pytest.mark.parametrize(
    "gain_aux_db, snr_db",
    [
        (0, None),
        (10, None),
        (None, -10),
        (None, 0),
        (None, 10),
        (0, 10),
    ],
)
def test_mix_2(
    base_duration,
    aux_duration,
    write_pos_base,
    extend_base,
    read_pos_aux,
    read_dur_aux,
    loop_aux,
    gain_base_db,
    gain_aux_db,
    snr_db,
):
    aux = np.array([range(aux_duration)])
    base_value = 0.1
    base = base_value * np.ones((1, base_duration))

    # To calculate the expected mix
    # we manipulate the aux signal
    # and store it in a different variable
    aux_expected = aux.copy()

    # Shrink aux to selected region
    if read_dur_aux is not None and read_dur_aux > 0:
        end = read_pos_aux + read_dur_aux
        aux_expected = aux_expected[:, read_pos_aux:end]
    else:
        aux_expected = aux_expected[:, read_pos_aux:]

    # Expand aux signal for loop_aux
    # to simulate looping
    has_looped = False
    if loop_aux and aux_expected.shape[1] < base_duration:
        aux_expected = np.concatenate([aux_expected] * 3, axis=1)
        has_looped = True

    # Number of samples available for mix in base
    len_mix_base = base.shape[1] - write_pos_base

    # Number of samples read for mix from aux
    if read_dur_aux is None or read_dur_aux == 0:
        len_mix_aux = max(
            [
                aux_expected.shape[1],
                len_mix_base,
            ]
        )
    else:
        len_mix_aux = read_dur_aux

    # If number of samples available for mix in base
    # is smaller than the number of samples read from aux
    # we pad zeros to base if extend_base is `True`.
    # Otherwise we trim the aux signal.
    gain_base = auglib.utils.from_db(gain_base_db)
    expected_mix = gain_base * base_value * np.ones(base.shape)
    if len_mix_aux > len_mix_base:
        if (
            extend_base
            and not has_looped
            or (
                extend_base
                and read_dur_aux is not None
                and read_dur_aux > base_duration
            )
        ):
            expected_mix = np.concatenate(
                [expected_mix, np.zeros((1, len_mix_aux - len_mix_base))],
                axis=1,
            )
        else:
            len_mix_aux = len_mix_base

    # read_dur_aux is allowed to extend aux signal,
    # in this case zeros are padded at the end.
    # Those zeros will NOT be included in the RMS_dB calculation
    len_mix_aux = min(
        aux_expected.shape[1],
        len_mix_aux,
    )  # this seems to be not needed
    # len_mix_aux = len_mix_aux - write_pos_base
    if loop_aux and write_pos_base > 0 and base_duration > aux_duration:
        len_mix_aux = min([len_mix_aux, len_mix_base])
    aux_expected = aux_expected[:, :len_mix_aux]

    # As we use a fixed signal for `base`
    # the RMS_db value is independent of signal length
    rms_base_start = write_pos_base
    rms_base_end = min([base_duration, aux_expected.shape[1] + write_pos_base])
    rms_aux_end = min(base_duration - write_pos_base, aux_expected.shape[1])

    rms_db_base = audmath.db(  # includes already gain_base_db
        audmath.rms(expected_mix[:, rms_base_start:rms_base_end])
    )
    rms_db_aux = audmath.db(audmath.rms(aux_expected[:, :rms_aux_end]))

    # Get gain factor for aux
    if gain_aux_db is None:
        gain_aux_db = 0.0
    if snr_db is not None:
        gain_aux_db = rms_db_base - snr_db - rms_db_aux
    gain_aux = auglib.utils.from_db(gain_aux_db)

    # Add aux values to expected mix
    mix_start = write_pos_base
    expected_mix[:, mix_start : mix_start + len_mix_aux] += gain_aux * aux_expected

    expected_mix = expected_mix.astype(auglib.core.transform.DTYPE)
    transform = auglib.transform.Mix(
        aux,
        gain_base_db=gain_base_db,
        gain_aux_db=gain_aux_db,
        snr_db=snr_db,
        write_pos_base=write_pos_base,
        read_pos_aux=read_pos_aux,
        read_dur_aux=read_dur_aux,
        extend_base=extend_base,
        unit="samples",
        loop_aux=loop_aux,
    )
    augmented_signal = transform(base)
    assert augmented_signal.dtype == expected_mix.dtype
    assert augmented_signal.shape == expected_mix.shape
    np.testing.assert_almost_equal(
        augmented_signal,
        expected_mix,
        decimal=5,
    )


@pytest.mark.parametrize("unit", ["relative"])
@pytest.mark.parametrize(
    "base, aux, write_pos_base, read_pos_aux, read_dur_aux, expected",
    [
        (
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([1, 2, 3, 4]),
            0.5,
            0.5,
            0.5,
            np.array([0.1, 0.1, 3.1, 4.1], dtype="float32"),
        ),
    ],
)
def test_mix_3(
    unit,
    base,
    aux,
    write_pos_base,
    read_pos_aux,
    read_dur_aux,
    expected,
):
    transform = auglib.transform.Mix(
        aux,
        write_pos_base=write_pos_base,
        read_pos_aux=read_pos_aux,
        read_dur_aux=read_dur_aux,
        unit=unit,
    )
    np.testing.assert_array_equal(
        transform(base),
        expected,
        strict=True,
    )
