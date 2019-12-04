import numpy as np
import pytest
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
