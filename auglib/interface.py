import os
import glob
import pandas as pd
import audiofile as af
from .buffer import AudioBuffer, Transform


class OnlineTransform(object):
    def __init__(self, transform: Transform, sampling_rate: int):
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __call__(self, signal):
        with AudioBuffer.from_array(signal, self.sampling_rate) as buf:
            self.transform(buf)
            transformed = buf.to_array()
        return transformed


class OfflineTransform(object):
    def __init__(self, transform: Transform,
                 precision: str = '16bit', normalize: bool = False):
        self.transform = transform

    def apply_on_filename(self, input_filename: str, target_filename: str, *,
                          offset: int = 0, duration: int = None):
        with AudioBuffer.read(
                input_filename, offset=offset, duration=duration) as buf:
            self.transform(buf)
            buf.write(target_filename)

    def apply_on_folder(self, input_folder, output_folder):
        supported_formats = ['wav', 'ogg', 'flac']
        files = []
        for format in supported_formats:
            files += glob.glob(input_folder + '/*.' + format)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in files:
            self.apply_on_filename(
                filename, os.path.join(
                    output_folder, os.path.basename(filename)))

    def apply_on_index(self, index: pd.MultiIndex, output_folder: str, *,
                       force_overwrite: bool = False):
        df = index.to_frame(index=False).copy()
        index_columns = list(df.columns)
        augmented_filename = os.path.join(output_folder, 'augmented.pkl')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        elif os.path.isfile(augmented_filename) and not force_overwrite:
            return pd.read_pickle(augmented_filename)

        df_augmented = df.copy()
        for index, row in df.iterrows():
            duration = None
            offset = 0
            if ('start' in df.columns) and ('end' in df.columns):
                offset = row['start'].total_seconds()
                if not pd.isnull(row['end']):
                    duration = row['end'].total_seconds() - offset
            self.apply_on_filename(
                row['file'],
                os.path.join(output_folder, os.path.basename(row['file'])),
                offset=offset,
                duration=duration
            )
        df_augmented['augmented_file'] = df_augmented['file'].apply(
            lambda x: os.path.join(output_folder, os.path.basename(x)))
        df_augmented.set_index(index_columns).to_pickle(augmented_filename)
        return df_augmented
