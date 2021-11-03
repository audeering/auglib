import filecmp
import os

import audiofile
import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audiofile as af
import audobject

import auglib


def map_files(index, root):
    if audformat.index_type(index) == audformat.define.IndexType.SEGMENTED:
        files = index.levels[0]
        files = [os.path.join(root, file) for file in files]
        return index.set_levels(files, level=0)
    else:
        return index.map(lambda x: os.path.join(root, x))


@pytest.mark.parametrize(
    'index, signal, sampling_rate, transform, keep_nat, '
    'expected_index, expected_signal',
    [
        # segment length unchanged
        (
            audformat.filewise_index(
                ['f1.wav', 'f2.wav'],
            ),
            np.zeros((1, 10)),
            10,
            auglib.transform.Function(lambda x, _: x + 1),
            False,
            audformat.segmented_index(
                ['f1.wav', 'f2.wav'],
                [0, 0],
                ['1s', '1s'],
            ),
            np.ones((1, 10)),
        ),
        (
            audformat.filewise_index(
                ['f1.wav', 'f2.wav'],
            ),
            np.zeros((1, 10)),
            10,
            auglib.transform.Function(lambda x, _: x + 1),
            True,
            audformat.segmented_index(
                ['f1.wav', 'f2.wav'],
                [0, 0],
                [pd.NaT, pd.NaT],
            ),
            np.ones((1, 10)),
        ),
        (
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.8s'],
                ['0.2s', '0.9s'],
            ),
            np.zeros((1, 10)),
            10,
            auglib.transform.Function(lambda x, _: x + 1),
            False,
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.8s'],
                ['0.2s', '0.9s'],
            ),
            np.array([[0., 1., 0., 0., 0., 0., 0., 0., 1., 0.]]),
        ),
        # expand segments
        (
            audformat.filewise_index(
                ['f1.wav', 'f2.wav'],
            ),
            np.zeros((1, 5)),
            10,
            auglib.transform.Compose(
                [
                    auglib.transform.Function(lambda x, _: x + 1),
                    auglib.transform.AppendValue(5, 2, unit='samples'),
                ]
            ),
            False,
            audformat.segmented_index(
                ['f1.wav', 'f2.wav'],
                [0, 0],
                ['1s', '1s'],
            ),
            np.array([[1., 1., 1., 1., 1., 2., 2., 2., 2., 2.]]),
        ),
        (
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.8s'],
                ['0.2s', '0.9s'],
            ),
            np.zeros((1, 10)),
            10,
            auglib.transform.Compose(
                [
                    auglib.transform.Function(lambda x, _: x + 1),
                    auglib.transform.AppendValue(1, 2, unit='samples'),
                ]
            ),
            False,
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.9s'],
                ['0.3s', '1.1s'],
            ),
            np.array([[0., 1., 2., 0., 0., 0., 0., 0., 0., 1., 2., 0.]]),
        ),
        # trim segments
        (
            audformat.filewise_index(
                ['f1.wav', 'f2.wav'],
            ),
            np.zeros((1, 10)),
            10,
            auglib.transform.Compose(
                [
                    auglib.transform.Function(lambda x, _: x + 1),
                    auglib.transform.Trim(duration=5, unit='samples'),
                ]
            ),
            False,
            audformat.segmented_index(
                ['f1.wav', 'f2.wav'],
                [0, 0],
                ['0.5s', '0.5s'],
            ),
            np.ones((1, 5)),
        ),
        (
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.6s'],
                ['0.4s', '0.9s'],
            ),
            np.zeros((1, 10)),
            10,
            auglib.transform.Compose(
                [
                    auglib.transform.Function(lambda x, _: x + 1),
                    auglib.transform.Trim(duration=2, unit='samples'),
                ]
            ),
            False,
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.5s'],
                ['0.3s', '0.7s'],
            ),
            np.array([[0., 1., 1., 0., 0., 1., 1., 0.]]),
        ),
    ]
)
def test_augment_new(tmpdir, index, signal, sampling_rate, transform,
                     keep_nat, expected_index, expected_signal):

    # create interface

    augment = auglib.Augment(
        transform,
        sampling_rate=sampling_rate,
        keep_nat=keep_nat,
    )

    # create input files and expand path

    root = os.path.join(tmpdir, 'input')
    cache_root = os.path.join(tmpdir, 'cache')
    expected_root = os.path.join(cache_root, augment.short_id, '0')

    index = map_files(index, root)
    files = index.get_level_values('file').unique()
    for file in files:
        audeer.mkdir(os.path.dirname(file))
        audiofile.write(file, signal, sampling_rate)
    expected_index = map_files(expected_index, expected_root)

    # augment index

    augmented_index = augment.augment(
        index,
        cache_root=cache_root,
        remove_root=root,
        force=True,
    )
    pd.testing.assert_index_equal(augmented_index, expected_index)

    expected_files = augmented_index.get_level_values('file').unique()
    for file in expected_files:
        tmp_file = os.path.join(tmpdir, 'tmp.wav')
        audiofile.write(tmp_file, expected_signal, sampling_rate)
        filecmp.cmp(file, tmp_file)

    # augment series

    y = pd.Series(0.0, index=index)
    augmented_y = augment.augment(
        y,
        cache_root=cache_root,
        remove_root=root,
        force=True,
    )
    expected_y = y.set_axis(expected_index)
    pd.testing.assert_series_equal(augmented_y, expected_y)

    # augment frame

    df = pd.DataFrame({'a': 0.0, 'b': 1.0}, index=index)
    augmented_df = augment.augment(
        df,
        cache_root=cache_root,
        remove_root=root,
        force=True,
    )
    expected_df = df.set_axis(expected_index)
    pd.testing.assert_frame_equal(augmented_df, expected_df)


def test_augment_empty(tmpdir):

    data = pd.Series(
        None,
        index=audformat.segmented_index(),
        dtype='float64',
    )
    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
    )
    result = process.augment(data, cache_root=tmpdir)
    assert result.empty


@pytest.mark.parametrize(
    'signal',
    [
        np.array(
            [
                [1., 1., 1., 1.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
            ],
            dtype='float32',
        )
    ]
)
@pytest.mark.parametrize(
    'transform, expected',
    [
        (
            auglib.transform.Function(lambda x, _: x + 1),
            np.array(
                [
                    [2., 2., 2., 2.],
                    [3., 3., 3., 3.],
                    [4., 4., 4., 4.],
                ],
                dtype='float32',
            )
        ),
        (
            auglib.transform.AppendValue(
                duration=2,
                unit='samples',
            ),
            np.array(
                [
                    [1., 1., 1., 1., 0., 0.],
                    [2., 2., 2., 2., 0., 0.],
                    [3., 3., 3., 3., 0., 0.],
                ],
                dtype='float32',
            )
        ),
        (
            auglib.transform.Trim(
                duration=auglib.observe.List([3, 2, 1]),
                unit='samples',
            ),
            np.array(
                [
                    [1., 1., 1.],
                    [2., 2., 0.],
                    [3., 0., 0.],
                ],
                dtype='float32',
            )
        )
    ]
)
def test_augment_multichannel(signal, transform, expected):
    augment = auglib.Augment(transform)
    signal_augmented = augment(signal, 8000)
    np.testing.assert_equal(signal_augmented, expected)


def test_augment_num_workers(tmpdir):

    # create dummy signal and interface

    files = [f'f{idx}.wav' for idx in range(15)]
    index = audformat.filewise_index(files)
    signal = np.zeros((1, 10))
    sampling_rate = 10
    transform = auglib.transform.Function(lambda x, _: x + 1)

    # create input files

    root = os.path.join(tmpdir, 'input')
    cache_root = os.path.join(tmpdir, 'cache')

    index = map_files(index, root)
    files = index.get_level_values('file').unique()
    for file in files:
        audeer.mkdir(os.path.dirname(file))
        audiofile.write(file, signal, sampling_rate)

    # single thread

    augment = auglib.Augment(
        transform,
        num_workers=1,
    )
    y_single = augment.augment(
        index,
        cache_root=cache_root,
        force=True,
    )

    # multiple threads

    augment = auglib.Augment(
        transform,
        num_workers=5,
    )
    y_multi = augment.augment(
        index,
        cache_root=cache_root,
        force=True,
    )

    pd.testing.assert_index_equal(y_single, y_multi)


@pytest.mark.parametrize(
    'remove_root',
    [
        None,
        pytest.AUDB_ROOT,
        pytest.param(
            '/invalid/directory', marks=pytest.mark.xfail(raises=RuntimeError)
        ),
        pytest.param(
            pytest.AUDB_ROOT[:len(pytest.AUDB_ROOT) - 1],
            marks=pytest.mark.xfail(raises=RuntimeError)
        )
    ]
)
def test_augment_remove_root(tmpdir, remove_root):

    data = pytest.DATA_FILES
    original_file = data[0]
    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
    )
    result = process.augment(
        data,
        cache_root=tmpdir,
        remove_root=remove_root,
    )
    augmented_file = result.levels[0][0]
    if remove_root is None:
        augmented_file.endswith(original_file)
    else:
        augmented_file.endswith(original_file.replace(remove_root, ''))


@pytest.mark.parametrize(
    'sampling_rate, target_rate, resample, modified_only',
    [
        (
            10, 10, False, True,
        ),
        (
            10, 20, True, True,
        ),
        (
            10, 5, True, True,
        ),
        pytest.param(  # sampling rate mismatch
            10, 20, False, True,
            marks=pytest.mark.xfail(raises=RuntimeError)
        ),
        pytest.param(  # resampling with modified_only=False
            10, 20, True, False,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
    ]
)
def test_augment_resample(tmpdir, sampling_rate, target_rate, resample,
                          modified_only):

    # create dummy signal and interface

    index = audformat.filewise_index(['f1.wav', 'f2.wav'])
    signal = np.zeros((1, 10))
    transform = auglib.transform.Function(lambda x, _: x + 1)
    augment = auglib.Augment(
        transform,
        resample=resample,
        sampling_rate=target_rate,
    )

    # create input files

    root = os.path.join(tmpdir, 'input')
    cache_root = os.path.join(tmpdir, 'cache')

    index = map_files(index, root)
    files = index.get_level_values('file').unique()
    for file in files:
        audeer.mkdir(os.path.dirname(file))
        audiofile.write(file, signal, sampling_rate)

    # augment index

    augmented_index = augment.augment(
        index,
        cache_root=cache_root,
        modified_only=modified_only,
        remove_root=root,
    )
    augmented_files = augmented_index.get_level_values('file').unique()
    for augmented_file in augmented_files:
        assert audiofile.sampling_rate(augmented_file) == target_rate


def test_augment_seed():

    sr = 8000
    x = np.zeros((1, 8))
    transform = auglib.transform.PinkNoise()

    for seed in [None, 1]:

        auglib.seed(0)

        augment = auglib.Augment(transform, seed=seed, num_workers=5)
        augment_yaml = augment.to_yaml_s()
        y = augment(x, sr)

        augment_2 = audobject.from_yaml_s(augment_yaml)
        y_2 = augment_2(x, sr)  # matches y if seed == 1

        augment_3 = audobject.from_yaml_s(augment_yaml, seed=0)
        y_3 = augment_3(x, sr)  # matches y if seed == None

        augment_4 = audobject.from_yaml_s(augment_yaml, seed=None)
        y_4 = augment_4(x, sr)  # never matches y

        if seed is None:
            assert augment.num_workers == 5
            with pytest.raises(AssertionError):
                np.testing.assert_equal(y, y_2)
            np.testing.assert_equal(y, y_3)
        else:
            assert augment.num_workers == 1
            np.testing.assert_equal(y, y_2)
            with pytest.raises(AssertionError):
                np.testing.assert_equal(y, y_3)

        with pytest.raises(AssertionError):
            np.testing.assert_equal(y, y_4)


@pytest.mark.parametrize(
    'index, num_variants, modified_only, keep_nat',
    [
        (
            audformat.filewise_index(['f1.wav', 'f2.wav']),
            1,
            True,
            False,
        ),
        (
            audformat.segmented_index(
                ['f1.wav', 'f1.wav'],
                ['0.1s', '0.8s'],
                ['0.2s', '0.9s'],
            ),
            3,
            True,
            False,
        ),
        (
            audformat.filewise_index(['f1.wav', 'f2.wav']),
            3,
            False,
            False,
        ),
        (
            audformat.filewise_index(['f1.wav', 'f2.wav']),
            3,
            False,
            True,
        ),
    ]
)
def test_augment_variants(tmpdir, index, num_variants, modified_only,
                          keep_nat):

    # create dummy signal and interface

    signal = np.zeros((1, 10))
    sampling_rate = 10
    transform = auglib.transform.Function(lambda x, _: x + 1)
    augment = auglib.Augment(transform, keep_nat=keep_nat)

    # list with expected files

    root = os.path.join(tmpdir, 'input')
    cache_root = os.path.join(tmpdir, 'cache')

    files = index.get_level_values('file').unique()
    expected_files = []
    for idx in range(num_variants):
        cache_root_idx = os.path.join(
            cache_root,
            augment.short_id,
            str(idx),
        )
        for file in files:
            expected_files.append(os.path.join(cache_root_idx, file))

    # create input files

    index = map_files(index, root)
    files = index.get_level_values('file').unique()
    for file in files:
        audeer.mkdir(os.path.dirname(file))
        audiofile.write(file, signal, sampling_rate)

    if not modified_only:
        expected_files = files.tolist() + expected_files

    # augment index

    augmented_index = augment.augment(
        index,
        cache_root=cache_root,
        num_variants=num_variants,
        modified_only=modified_only,
        remove_root=root,
    )

    augmented_files = augmented_index.get_level_values('file').unique()
    assert augmented_files.tolist() == expected_files


def test_cache_root(tmpdir):

    auglib.config.CACHE_ROOT = tmpdir

    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
    )
    process_root = os.path.join(tmpdir, process.short_id)
    result = process.augment(pytest.DATA_FILES)
    result[0][0].startswith(str(tmpdir))

    assert auglib.default_cache_root() == tmpdir
    assert auglib.default_cache_root(process) == process_root
    assert len(audeer.list_file_names(process_root)) > 0

    auglib.clear_default_cache_root(process)
    assert os.path.exists(auglib.default_cache_root())
    assert not os.path.exists(process_root)

    auglib.clear_default_cache_root()
    assert os.path.exists(auglib.default_cache_root())
