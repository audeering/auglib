import glob
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import audata
import audata.testing
import audiofile as af

import auglib
from auglib import NumpyTransform, AudioModifier
from auglib.transform import NormalizeByPeak, AppendValue


@pytest.mark.parametrize('n,sr',
                         [(10, 8000),
                          (10, 44100)])
def test_numpytransform(n, sr):

    x = np.linspace(-0.5, 0.5, num=n)
    t = NumpyTransform(NormalizeByPeak(), sr)
    assert np.abs(t(x)).max() == 1.0


@pytest.mark.parametrize('duration', [(5)])
def test_audiomodifier_apply_on_index(duration):
    db = audata.testing.create_db(minimal=True)
    db.schemes['anger'] = audata.Scheme('int', minimum=1, maximum=5)
    audata.testing.add_table(
        db,
        table_id='anger',
        table_type=audata.define.TableType.FILEWISE,
        columns='anger'
    )
    audata.testing.create_audio_files(db, root='./')
    df = db['anger'].df

    transform = AppendValue(duration)
    t = AudioModifier(transform)
    df_augmented = t.apply_on_index(df.index, 'augmented')
    assert len(df_augmented) == len(df)

    for index, row in df_augmented.to_frame().reset_index().iterrows():
        assert af.duration(row['augmented_file']) == af.duration(
            row['file']) + duration

    shutil.rmtree('audio')
    shutil.rmtree('augmented')


@pytest.mark.parametrize('duration', [(5)])
def test_audiomodifier_apply_on_database(duration):
    db = audata.testing.create_db(minimal=True)
    db.schemes['anger'] = audata.Scheme('int', minimum=1, maximum=5)
    audata.testing.add_table(
        db,
        table_id='anger',
        table_type=audata.define.TableType.FILEWISE,
        columns='anger'
    )
    audata.testing.create_audio_files(db, root='db')
    db.save('db')

    transform = AppendValue(duration)
    t = AudioModifier(transform)
    t.apply_on_database('db', 'augmented')

    db = audata.Database.load('db')
    db.map_files(lambda x: os.path.abspath(os.path.join('db', x)))
    augmented = audata.Database.load('augmented')
    augmented.map_files(os.path.abspath)

    df = db['anger'].df
    augmented_df = augmented['anger'].df

    assert len(df) == len(augmented_df)

    for index, row in augmented_df.reset_index().iterrows():
        original_duration = af.duration(
            df.index.get_level_values('file')[index])
        augmented_duration = af.duration(row['file'])
        assert augmented_duration == original_duration + duration

    shutil.rmtree('db')
    shutil.rmtree('augmented')


@pytest.mark.parametrize('duration', [(5)])
def test_audiomodifier_apply_on_folder(duration):
    db = audata.testing.create_db(minimal=True)
    db.schemes['anger'] = audata.Scheme('int', minimum=1, maximum=5)
    audata.testing.add_table(
        db,
        table_id='anger',
        table_type=audata.define.TableType.FILEWISE,
        columns='anger'
    )
    audata.testing.create_audio_files(db, root='./')

    transform = AppendValue(duration)
    t = AudioModifier(transform)
    t.apply_on_folder('audio', 'augmented')

    clean_files = glob.glob('audio' + '/*.wav')
    augmented_files = glob.glob('augmented' + '/*.wav')

    assert len(clean_files) == len(augmented_files)

    for augmented_file in augmented_files:
        assert af.duration(augmented_file) == af.duration(
            os.path.join('audio', os.path.basename(augmented_file))) + duration

    shutil.rmtree('audio')
    shutil.rmtree('augmented')


@pytest.mark.parametrize('duration', [(5)])
def test_audiomodifier_apply_on_file(duration):
    initial_dur = 1  # in seconds
    sr = 16000
    dummy_audio = np.zeros(initial_dur * sr)
    af.write('test.wav', dummy_audio, sr)
    transform = AppendValue(duration)
    t = AudioModifier(transform)
    t.apply_on_file('test.wav', 'augmented.wav')
    assert af.duration('augmented.wav') == af.duration('test.wav') + duration
    os.remove('test.wav')
    os.remove('augmented.wav')


@pytest.mark.parametrize(
    'sampling_rate, resample',
    [
        (
            None, False,
        ),
        (
            8000, True,
        ),
        pytest.param(
            8000, False,
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
    'data, modified_only, num_variants',
    [
        (
            pytest.DATA_FILES,
            True,
            1,
        ),
        (
            pytest.DATA_COLUMN,
            False,
            1,
        ),
        (
            pytest.DATA_TABLE,
            True,
            3,
        ),
    ]
)
def test_augment(tmpdir, sampling_rate, resample, keep_nat,
                 num_workers, data, modified_only, num_variants):

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
            tmpdir,
            modified_only=modified_only,
            num_variants=num_variants,
            force=force,
        )

        expected = []

        for idx in range(num_variants):

            cache_root_idx = process._safe_cache(tmpdir, idx)
            segmented = audata.utils.to_segmented_frame(data)

            if not modified_only and idx == 0:
                if not keep_nat:
                    files = segmented.index.get_level_values('file')
                    starts = segmented.index.get_level_values('start')
                    ends = segmented.index.get_level_values('end')
                    ends = [
                        pd.to_timedelta(af.duration(file), 's')
                        if pd.isna(end) else end
                        for file, end in zip(files, ends)
                    ]
                    index = pd.MultiIndex.from_arrays(
                        [files, starts, ends],
                        names=['file', 'start', 'end'],
                    )
                    new_data = segmented.copy()
                    new_data.index = index
                    expected.append(new_data)
                else:
                    expected.append(segmented)

            files = auglib.Augment._out_files(
                segmented.index.get_level_values('file'),
                cache_root_idx,
            )
            starts = segmented.index.get_level_values('start')
            ends = segmented.index.get_level_values('end')
            if not keep_nat:
                ends = [
                    pd.to_timedelta(af.duration(file), 's')
                    if pd.isna(end) else end
                    for file, end in zip(files, ends)
                ]
            index = pd.MultiIndex.from_arrays(
                [files, starts, ends],
                names=['file', 'start', 'end'],
            )
            new_data = segmented.copy()
            new_data.index = index
            expected.append(new_data)

        expected = pd.concat(expected, axis='index')
        if isinstance(result, pd.Series):
            pd.testing.assert_series_equal(expected, result)
        else:
            pd.testing.assert_frame_equal(expected, result)

        # load augmented file and test if segments are set to 1
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
        index=pd.MultiIndex.from_arrays(
            [[], [], []],
            names=['file', 'start', 'end'],
        )
    )
    transform = pytest.TRANSFORM_ONES
    process = auglib.Augment(
        transform=transform,
    )
    result = process.augment(data, tmpdir)
    assert result.empty
