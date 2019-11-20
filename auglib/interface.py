import os
import glob
import warnings
from typing import Union, Sequence
import tqdm
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from audata import Database

from .buffer import AudioBuffer, Transform


# TODO: replace with functions from audeer

def _safe_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _remove(path: str):
    path = _safe_path(path)
    if os.path.exists(path):
        os.remove(path)


def _make_dirs(path: str):
    path = _safe_path(path)
    if not os.path.exists(path):
        os.makedirs(path)


def _make_tree(files: Sequence[str]):
    dirs = set()
    for f in files:
        dirs.add(os.path.dirname(f))
    for d in list(dirs):
        _make_dirs(d)


class NumpyTransform(object):
    r"""Interface for data augmentation on numpy arrays.

    Wraps a :class:`auglib.buffer.Transform` under a ``Callable``
    interface, allowing it to be called on-the-fly on numpy arrays.

    Args:
        transform: auglib transform to be wrapped
        sampling_rate: sampling rate of incoming signals

    """

    def __init__(self, transform: Transform, sampling_rate: int):
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        with AudioBuffer.from_array(signal, self.sampling_rate) as buf:
            self.transform(buf)
            transformed = buf.to_array()
        return transformed


class AudioModifier(object):
    r"""Interface for modifying audio offline.

    Provides utility functions for :class:`auglib.buffer.Transform`. Enables
    it to work on:

    * files
    * folders
    * :class:`pandas.Index` coming from the `Unified Format`_
    * :class:`audata.Database` in the `Unified Format`_

    Augments files and then stores it on disk for future usage.

    Args:
        transform: transform to be applied on incoming data
        precision: precision to store output data in
        normalize: controls whether output audio should be normalized

    .. _Unified Format: http://tools.pp.audeering.com/audata/data-format
        .html

    """

    def __init__(self, transform: Transform,
                 precision: str = '16bit',
                 normalize: bool = False):
        self.transform = transform
        self.precision = precision
        self.normalize = normalize

    def apply_on_file(self, input_file: str, output_file: str, *,
                      start: pd.Timedelta = None, end: pd.Timedelta = None):
        r"""Applies transform on input file and writes output to output file.

        Args:
            input_file: file to be augmented
            output_file: output file
            start: start position
            end: end position

        """
        input_file = _safe_path(input_file)
        output_file = _safe_path(output_file)
        start = start or pd.NaT
        end = end or pd.NaT

        offset = start.total_seconds() if not pd.isna(start) else 0
        duration = end.total_seconds() - offset if not pd.isna(end) else None

        with AudioBuffer.read(
                input_file, offset=offset, duration=duration) as buf:
            self.transform(buf)
            buf.write(output_file, precision=self.precision,
                      normalize=self.normalize)

    def _apply_on_file_job(self, arg):
        idx, file_input, file_output, start, end = arg
        self.apply_on_file(file_input, file_output, start=start, end=end)

    def apply_on_files(self, input_files: Sequence[str],
                       output_files: Sequence[str], *,
                       starts: Sequence[pd.Timedelta] = None,
                       ends: Sequence[pd.Timedelta] = None,
                       num_jobs: int = None,
                       verbose: bool = True):
        r"""Applies transform on a list of input files.

        Args:
            input_files: list of files to be augmented
            output_files: list of output files
            starts: list of start positions
            ends: list of end positions
            num_jobs: number of parallel jobs. If ``None`` is set to number
                of available CPUs
            verbose: show debug messages

        """

        n = len(input_files)

        if not n == len(output_files):
            raise ValueError('lists with input and output files must have the '
                             'same length')

        if starts is None:
            starts = [pd.NaT] * n
        if ends is None:
            ends = [pd.NaT] * n

        if not n == len(starts):
            raise ValueError('lists with start positions and files must be of '
                             'same length')
        if not n == len(ends):
            raise ValueError('lists with end positions and files must be of '
                             'same length')

        args = [(idx, file_input, file_output, start, end)
                for idx, (file_input, file_output, start, end) in
                enumerate(zip(input_files, output_files, starts, ends))]
        num_jobs = num_jobs or cpu_count()

        with Pool(num_jobs) as pool:
            for _, result in tqdm.tqdm(
                    enumerate(pool.imap(self._apply_on_file_job, args)),
                    desc='Modify', total=n, disable=not verbose):
                pass

    def apply_on_folder(self, input_folder: str, output_folder: str, *,
                        num_jobs: int = None, verbose: bool = True):
        r"""Applies transform on all files in a folder.

        .. note:: Looks for all files in the supported formats
            (``'wav'``, ``'ogg'``, ``'flac'``)

        Args:
            input_folder: folder to search for files to augment
            output_folder: folder to store augmented data in
            num_jobs: number of parallel jobs. If ``None`` is set to number
                of available CPUs
            verbose: show debug messages

        """
        input_folder = _safe_path(input_folder)
        output_folder = _safe_path(output_folder)

        supported_formats = ['wav', 'ogg', 'flac']
        input_files = []
        for f in supported_formats:
            input_files += glob.glob(input_folder + '/*.' + f)
        output_files = [os.path.join(output_folder, os.path.basename(f))
                        for f in input_files]

        _make_dirs(output_folder)

        self.apply_on_files(input_files, output_files, num_jobs=num_jobs,
                            verbose=verbose)

    def apply_on_index(self, index: Union[pd.Index, pd.DataFrame],
                       output_folder: str, *,
                       window_size: pd.Timedelta = None,
                       force_overwrite: bool = False,
                       num_jobs: int = None, verbose: bool = True)\
            -> pd.Series:
        r"""Applies transform on data indexed in a :class:`pandas.DataFrame`.

        This is intended for usage with data coming from the `Unified Format`_.
        Returns a :class:`pandas.Series` containing identical index pointing
        to the generated augmented files.

        .. note:: If a previous version is found, it checks if
            index or transform has changed and possibly rerun the job.

        Args:
            index: index of a table in the `Unified Format`_
            output_folder: folder to store data in
            window_size: custom windows size if index is continuous
            force_overwrite: if set to ``True`` then **all** in the output
                folder will be removed, otherwise the old data will be returned
            num_jobs: number of parallel jobs. If ``None`` is set to number
                of available CPUs
            verbose: show debug messages

        Example:
            >>> import audb, audata, auglib
            >>> db = audb.load('emodb')
            >>> df = audata.Filter('emotion')(db)
            >>> transform = auglib.transform.AppendValue(5.0)
            >>> t = AudioModifier(transform)
            >>> t.apply_on_index(df.index, './augmented')

        .. _Unified Format: http://tools.pp.audeering.com/audata/data-format
            .html

        """
        if len(output_folder) == 0:
            raise ValueError('Please provide a valid string as the '
                             'output folder.')

        output_folder = _safe_path(output_folder)
        mapping_file = os.path.join(output_folder, 'augmented.pkl')
        mapping_file_csv = mapping_file[:-4] + '.csv'
        transform_file = os.path.join(output_folder, 'augmented.yaml')

        if os.path.isfile(mapping_file) and os.path.isfile(transform_file):
            old_mapping = pd.read_pickle(mapping_file)
            old_transform = Transform.load(transform_file)
            if not old_mapping.index.equals(index) \
                    or old_transform != self.transform:
                warnings.warn(UserWarning('index or transform has '
                                          'changed, force overwrite'))
                force_overwrite = True
            if not force_overwrite:
                return old_mapping
            else:
                for file in old_mapping:
                    _remove(file)

        _make_dirs(output_folder)

        if isinstance(index, pd.DataFrame):
            index = index.index

        n = len(index)
        input_files = [None] * n
        output_files = [None] * n
        starts = [pd.NaT] * n
        ends = [pd.NaT] * n
        for idx, row in enumerate(index):
            output_files[idx] = os.path.join(
                output_folder, 'augmented_{:06d}.wav'.format(idx + 1))
            if len(row) == 3:    # segmented
                input_files[idx], starts[idx], ends[idx] = row
            elif len(row) == 2:  # continuous
                input_files[idx], starts[idx] = row
                if window_size is None:
                    times = index.get_level_values(1)
                    window_size = times[1] - times[0]
                    warnings.warn('setting window size to {}s'
                                  .format(window_size))
                ends[idx] = starts[idx] + window_size
            else:                # filewise
                input_files[idx] = row
        self.apply_on_files(input_files, output_files, starts=starts,
                            ends=ends, num_jobs=num_jobs, verbose=verbose)

        mapping = pd.Series(data=output_files, index=index,
                            name='augmented_file')
        mapping.to_pickle(mapping_file)
        mapping.to_csv(mapping_file_csv, header=True)
        self.transform.dump(transform_file)

        return mapping

    def apply_on_database(self, input_folder: str, output_folder: str, *,
                          num_jobs: int = None, verbose: bool = True):
        r"""Creates a copy of a database in the `Unified Format`_.

        All files referenced in the database will be augmented.

        .. note:: The output folder must be empty.

        Args:
            input_folder: folder with database `Unified Format`_
            output_folder: folder to store the augmented database to
            num_jobs: number of parallel jobs. If ``None`` is set to number
                of available CPUs
            verbose: show debug messages

        .. _Unified Format: http://tools.pp.audeering.com/audata/data-format
            .html

        """
        input_folder = _safe_path(input_folder)
        output_folder = _safe_path(output_folder)

        db = Database.load(input_folder)

        _make_dirs(output_folder)
        if os.listdir(output_folder):
            raise ValueError('output folder is not empty')

        db.map_files(lambda x: x if os.path.isabs(x) else os.path.join(
            input_folder, x))
        input_files = db.files.copy()
        db.map_files(lambda x: x.replace(input_folder, output_folder))
        output_files = db.files
        _make_tree(output_files)

        db.save(output_folder)
        self.apply_on_files(input_files, output_files, num_jobs=num_jobs,
                            verbose=verbose)
