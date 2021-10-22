import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audiofile as af
import audobject

import auglib


@pytest.mark.parametrize(
    'sampling_rate, resample, modified_only',
    [
        (
            None, False, True,
        ),
        (
            8000, True, True,
        ),
        (
            None, False, False,
        ),
        pytest.param(  # resampling with modified_only=False
            8000, True, False,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(  # sampling rate mismatch
            8000, False, True,
            marks=pytest.mark.xfail(raises=RuntimeError)
        ),
    ]
)
@pytest.mark.parametrize(
    'keep_nat',
    [
        True, False,
    ]
)
@pytest.mark.parametrize(
    'num_workers',
    [
        1, 5,
    ]
)
@pytest.mark.parametrize(
    'data, num_variants',
    [
        (
            pytest.DATA_FILES,
            1,
        ),
        (
            pytest.DATA_COLUMN,
            1,
        ),
        (
            pytest.DATA_TABLE,
            3,
        ),
    ]
)
def test_augment(tmpdir, sampling_rate, resample, modified_only,
                 keep_nat, num_workers, data, num_variants):

    # create a transformation that sets buffer to 1
    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
        sampling_rate=sampling_rate,
        resample=resample,
        keep_nat=keep_nat,
        num_workers=num_workers,
    )

    for force in [False, True, False]:

        result = process.augment(
            data,
            cache_root=tmpdir,
            modified_only=modified_only,
            num_variants=num_variants,
            force=force,
        )

        expected = []

        if isinstance(data, pd.Index):
            data = data.to_series()

        for idx in range(num_variants):

            cache_root_idx = os.path.join(
                tmpdir, process.id, str(idx),
            )
            segmented = audformat.utils.to_segmented_index(data)
            index = segmented.index

            if not modified_only and idx == 0:
                if not keep_nat:
                    files = index.get_level_values('file')
                    starts = index.get_level_values('start')
                    ends = index.get_level_values('end')
                    ends = [
                        pd.to_timedelta(af.duration(file), 's')
                        if pd.isna(end) else end
                        for file, end in zip(files, ends)
                    ]
                    new_index = audformat.segmented_index(
                        files=files,
                        starts=starts,
                        ends=ends,
                    )
                    new_data = segmented.copy()
                    new_data.index = new_index
                    expected.append(new_data)
                else:
                    expected.append(segmented)

            files = auglib.Augment._out_files(
                index.get_level_values('file'),
                cache_root_idx,
            )
            starts = index.get_level_values('start')
            ends = index.get_level_values('end')
            if not keep_nat:
                ends = [
                    pd.to_timedelta(af.duration(file), 's')
                    if pd.isna(end) else end
                    for file, end in zip(files, ends)
                ]
            new_index = audformat.segmented_index(
                files=files,
                starts=starts,
                ends=ends,
            )
            new_data = segmented.copy()
            new_data.index = new_index
            expected.append(new_data)

        expected = pd.concat(expected, axis='index')
        if isinstance(result, pd.Series):
            pd.testing.assert_series_equal(expected, result)
        elif isinstance(result, pd.DataFrame):
            pd.testing.assert_frame_equal(expected, result)
        else:
            pd.testing.assert_index_equal(expected.index, result)

        # load augmented file and test if segments are set to 1
        if isinstance(result, pd.Index):
            result = result.to_series()
        augmented_file = result.index[-1][0]
        augmented_signal, augmented_signal_sr = af.read(augmented_file)
        for start, end in result.loc[augmented_file].index:
            if pd.isna(end):
                end = pd.to_timedelta(af.duration(augmented_file), 's')
            start_i = int(start.total_seconds() * augmented_signal_sr)
            end_i = int(end.total_seconds() * augmented_signal_sr) - 1
            np.testing.assert_almost_equal(
                augmented_signal[start_i:end_i],
                np.ones(end_i - start_i, dtype=np.float32),
                decimal=4,
            )


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


def test_cache_root(tmpdir):

    auglib.config.CACHE_ROOT = tmpdir

    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
    )
    process_root = os.path.join(tmpdir, process.id)
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
