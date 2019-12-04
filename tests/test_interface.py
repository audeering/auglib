import numpy as np
import pytest
import os
import glob
import audata
import shutil
import audata.testing
import audiofile as af
from auglib.transform import NormalizeByPeak, AppendValue
from auglib.interface import NumpyTransform, AudioModifier


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
        scheme_ids='anger'
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
        scheme_ids='anger'
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
        scheme_ids='anger'
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
    os.system('sox -n -r 16000 -c 1 test.wav trim 0.0 1.0')
    transform = AppendValue(duration)
    t = AudioModifier(transform)
    t.apply_on_file('test.wav', 'augmented.wav')
    assert af.duration('augmented.wav') == af.duration('test.wav') + duration
    os.remove('test.wav')
    os.remove('augmented.wav')
