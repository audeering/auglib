import os
import glob
import numpy as np
import pandas as pd
import audiofile as af
from .buffer import AudioBuffer, Transform


class NumpyTransform(object):
    r"""Interface for on-line data augmentation.

    Wraps a py:class:`pyaglib.buffer.Transform` under a ``Callable`` interface,
    allowing it to be called on-the-fly on numpy arrays.

    Args:
        transform: pyauglib transform to be wrapped
        sampling_rate: sampling rate of incoming signals

    """

    def __init__(self, transform: Transform, sampling_rate: int):
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __call__(self, signal: np.array) -> np.array:
        with AudioBuffer.from_array(signal, self.sampling_rate) as buf:
            self.transform(buf)
            transformed = buf.to_array()
        return transformed


class OfflineTransform(object):
    r"""Interface for offline data augmentation.

    Provides utility functions for py:class:`pyaglib.buffer.Transform`. Enables
    it to work on:
    * files
    * folders
    * ``DataFrame`` indices coming from the unified format

    Augments files and then stores it on disk for future usage.

    Args:
        transform: transform to be applied on incoming data
        precision: precision to store output data in
        normalize: controls whether output audio should be normalized

    """

    def __init__(self, transform: Transform,
                 precision: str = '16bit',
                 normalize: bool = False):
        self.transform = transform

    def apply_on_filename(self, input_filename: str, target_filename: str, *,
                          offset: int = 0, duration: int = None):
        r"""Applies transform on input filename.

        Writes output to target filename.

        Args:
            input_filename: file to be augmented
            target_filename: output file
            offset: start time for input filename
            duration: duration of segment for input filename

        """
        with AudioBuffer.read(
                input_filename, offset=offset, duration=duration) as buf:
            self.transform(buf)
            buf.write(target_filename)

    def apply_on_folder(self, input_folder: str, output_folder: str):
        r"""Applies transform on all files in a folder.

        Looks for all files in the supported formats.

        Args:
            input_folder: folder to search for files to augment
            output_folder: folder to store augmented data in

        """
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
                       force_overwrite: bool = False) -> pd.DataFrame:
        r"""Applies transform on all data indexed in a ``DataFrame``.

        This is intended for usage with data coming from the unified format.

        Args:
            index: index containing filenames, start and end times, as created
                in the unified format
            output_folder: folder to store data in
            force_overwrite: if set to ``True`` then any data in the output
                folder will be overwritten, otherwise the old data will be
                returned

        Returns:
            ``DataFrame`` containing identical indices with the input and
                pointing to the generated augmented files

        Example:
            >>> import audb, audata, auglib
            >>> db = audb.load('emodb')
            >>> df = audata.Filter('emotion')(db)
            >>> transform = auglib.transform.AppendValue(5.0)
            >>> t = OfflineTransform(transform)
            >>> t.apply_on_index(df.index, './augmented')

        """
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
