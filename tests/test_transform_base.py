import os

import numpy as np
import pytest

import audiofile
import audobject

import auglib


# Define transform without aux
class Transform(auglib.transform.Base):
    def __init__(self, bypass_prob, preserve_level):
        super().__init__(
            bypass_prob=bypass_prob,
            preserve_level=preserve_level,
        )

    def _call(self, base, *, sampling_rate=None):
        return base + 1


# Define transform with aux
class TransformAux(auglib.transform.Base):
    def __init__(self, aux, *, preserve_level=False, transform=None):
        super().__init__(
            preserve_level=preserve_level,
            aux=aux,
            transform=transform,
        )

    def _call(self, base, aux, *, sampling_rate=None):
        return base + aux


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize(
    "bypass_prob, preserve_level, base, expected",
    [
        (None, False, [1, 1], [2, 2]),
        (None, True, [1, 1], [1, 1]),
        (1, False, [0, 0], [0, 0]),
        (1, True, [0, 0], [0, 0]),
    ],
)
def test_base(sampling_rate, bypass_prob, preserve_level, base, expected):
    transform = Transform(bypass_prob, preserve_level)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    assert transform.bypass_prob == bypass_prob
    assert transform.preserve_level == preserve_level
    np.testing.assert_almost_equal(
        transform(np.array(base, dtype="float32")),
        np.array(expected, dtype="float32"),
        decimal=4,
    )


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("base", [[0, 0]])
@pytest.mark.parametrize("from_file", [True, False])
@pytest.mark.parametrize("observe", [True, False])
@pytest.mark.parametrize(
    "transform, preserve_level, aux, expected",
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
            [1.0e-06, 1.0e-06],
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
            [1.0e-06, 1.0e-06],
        ),
    ],
)
def test_base_aux(
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
    aux = np.array(aux, dtype="float32")
    if from_file:
        path = os.path.join(tmpdir, "test.wav")
        audiofile.write(path, aux, sampling_rate)
        aux = path
    if observe:
        aux = auglib.observe.List([aux])
    base_transform = TransformAux(
        aux,
        preserve_level=preserve_level,
        transform=transform,
    )
    if observe and not from_file:
        error_msg = (
            "Cannot serialize list if it contains an instance "
            "of <class 'numpy.ndarray'>. "
            "As a workaround, save signal to disk and add filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            base_transform.to_yaml_s(include_version=False)

    elif isinstance(aux, np.ndarray):
        error_msg = (
            "Cannot serialize an instance "
            "of <class 'numpy.ndarray'>. "
            "As a workaround, save signal to disk and pass filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            base_transform.to_yaml_s(include_version=False)
    else:
        base_transform = audobject.from_yaml_s(
            base_transform.to_yaml_s(include_version=False),
        )
    assert base_transform.bypass_prob is None
    assert base_transform.preserve_level == preserve_level
    # unless signal is read from file
    # we skip the following test
    # as we cannot serialize a signal,
    # which is required to calculate its ID
    if from_file:
        assert base_transform.aux == aux
    assert base_transform.transform == transform
    np.testing.assert_almost_equal(
        base_transform(np.array(base, dtype="float32")),
        np.array(expected, dtype="float32"),
        decimal=4,
    )


@pytest.mark.parametrize("sampling_rate", [8000])
@pytest.mark.parametrize("base", [[0, 0]])
@pytest.mark.parametrize(
    "aux, transform, expected",
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
    ],
)
def test_base_aux_transform(sampling_rate, base, aux, transform, expected):
    transform = TransformAux(aux, transform=transform)
    transform = audobject.from_yaml_s(
        transform.to_yaml_s(include_version=False),
    )

    np.testing.assert_almost_equal(
        transform(np.array(base, dtype="float32")),
        np.array(expected, dtype="float32"),
        decimal=4,
    )
