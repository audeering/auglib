import os

import numpy as np
import pytest

import audobject

import auglib


# Define transform without aux
class Transform(auglib.transform.Base):

    def __init__(self, bypass_prob, preserve_level):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )

    def _call(self, base):
        base._data = base._data + 1
        return base


# Define transform with aux
class TransformAux(auglib.transform.Base):

    def __init__(self, aux, *, preserve_level=False, transform=None):
        super().__init__(
            preserve_level=preserve_level,
            aux=aux,
            transform=transform,
        )

    def _call(self, base, aux):
        base._data = base._data + aux._data
        return base


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'bypass_prob, preserve_level, base, expected',
    [
        (None, False, [1, 1], [2, 2]),
        (None, True, [1, 1], [1, 1]),
        (1, False, [0, 0], [0, 0]),
        (1, True, [0, 0], [0, 0]),
    ],
)
def test_Base(sampling_rate, bypass_prob, preserve_level, base, expected):

    transform = Transform(bypass_prob, preserve_level)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
        assert transform.bypass_prob == bypass_prob
        assert transform.preserve_level == preserve_level
        transform(base_buf)
        np.testing.assert_almost_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
            decimal=4,
        )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[0, 0]])
@pytest.mark.parametrize('from_file', [True, False])
@pytest.mark.parametrize('observe', [True, False])
@pytest.mark.parametrize(
    'transform, preserve_level, aux, expected',
    [
        (
            None,
            False,
            [1, 1],
            [1, 1],
        ),
        (
            None,
            True,
            [1, 1],
            [1.e-06, 1.e-06],
        ),
        (
            auglib.transform.Function(lambda x, sr: x + 1),
            False,
            [1, 1],
            [2, 2],
        ),
        (
            auglib.transform.Function(lambda x, sr: x + 1),
            True,
            [1, 1],
            [1.e-06, 1.e-06],
        ),
    ],
)
def test_Base_aux(
        tmpdir,
        sampling_rate,
        base,
        from_file,
        observe,
        transform,
        preserve_level,
        aux,
        expected,
):

    with auglib.AudioBuffer.from_array(aux, sampling_rate) as aux_buf:
        if from_file:
            path = os.path.join(tmpdir, 'test.wav')
            aux_buf.write(path)
            aux_buf.free()
            aux_buf = path
        if observe:
            aux_buf = auglib.observe.List([aux_buf])
        base_transform = TransformAux(
            aux_buf,
            preserve_level=preserve_level,
            transform=transform,
        )
        if observe and not from_file:
            error_msg = (
                "Cannot serialize list if it contains an instance "
                "of <class 'auglib.core.buffer.AudioBuffer'>. "
                "As a workaround, save buffer to disk and add filename."
            )
            with pytest.raises(ValueError, match=error_msg):
                base_transform.to_yaml_s(include_version=False)

        elif isinstance(aux_buf, auglib.AudioBuffer):
            error_msg = (
                "Cannot serialize an instance "
                "of <class 'auglib.core.buffer.AudioBuffer'>. "
                "As a workaround, save buffer to disk and pass filename."
            )
            with pytest.raises(ValueError, match=error_msg):
                base_transform.to_yaml_s(include_version=False)
        else:
            base_transform = audobject.from_yaml_s(
                base_transform.to_yaml_s(include_version=False),
            )
        assert base_transform.bypass_prob is None
        assert base_transform.preserve_level == preserve_level
        # unless buffer is read from file
        # we skip the following test
        # as we cannot serialize a buffer,
        # which is required to calculate its ID
        if from_file:
            assert base_transform.aux == aux_buf
        assert base_transform.transform == transform
        with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
            base_transform(base_buf)
            np.testing.assert_almost_equal(
                base_buf._data,
                np.array(expected, dtype=np.float32),
                decimal=4,
            )


@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('base', [[0, 0]])
@pytest.mark.parametrize(
    'aux, transform, expected',
    [
        (
            auglib.transform.Function(lambda x, sr: x + 1),
            None,
            [1, 1],
        ),
        (
            auglib.transform.Function(lambda x, sr: x + 1),
            auglib.transform.Function(lambda x, sr: x + 1),
            [2, 2],
        ),
    ]
)
def test_Base_aux_transform(sampling_rate, base, aux, transform, expected):

    transform = TransformAux(aux, transform=transform)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    with auglib.AudioBuffer.from_array(base, sampling_rate) as base_buf:
        transform(base_buf)
        np.testing.assert_almost_equal(
            base_buf._data,
            np.array(expected, dtype=np.float32),
            decimal=4,
        )
