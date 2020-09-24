import os
import glob
from multiprocessing import Pool, cpu_count
import typing
import warnings

import numpy as np
import pandas as pd

import audata
import audeer
import audinterface
import audiofile

from auglib.core.buffer import AudioBuffer, Transform


def _remove(path: str):  # pragma: no cover
    path = audeer.safe_path(path)
    if os.path.exists(path):
        os.remove(path)


def _make_tree(files: typing.Sequence[str]):  # pragma: no cover
    dirs = set()
    for f in files:
        dirs.add(os.path.dirname(f))
    for d in list(dirs):
        audeer.mkdir(d)


class NumpyTransform(object):
    r"""Interface for data augmentation on numpy arrays.

    Wraps a :class:`auglib.Transform` under a ``Callable``
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


class AudioModifier(object):  # pragma: no cover
    r"""Interface for modifying audio offline.

    Provides utility functions for :class:`auglib.Transform`. Enables
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
        input_file = audeer.safe_path(input_file)
        output_file = audeer.safe_path(output_file)
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

    def apply_on_files(self, input_files: typing.Sequence[str],
                       output_files: typing.Sequence[str], *,
                       starts: typing.Sequence[pd.Timedelta] = None,
                       ends: typing.Sequence[pd.Timedelta] = None,
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
            desc = audeer.format_display_message('Augment', pbar=True)
            for _, result in audeer.progress_bar(
                    enumerate(pool.imap(self._apply_on_file_job, args)),
                    desc=desc, total=n, disable=not verbose):
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
        input_folder = audeer.safe_path(input_folder)
        output_folder = audeer.safe_path(output_folder)

        supported_formats = ['wav', 'ogg', 'flac']
        input_files = []
        for f in supported_formats:
            input_files += glob.glob(input_folder + '/*.' + f)
        output_files = [os.path.join(output_folder, os.path.basename(f))
                        for f in input_files]

        audeer.mkdir(output_folder)

        self.apply_on_files(input_files, output_files, num_jobs=num_jobs,
                            verbose=verbose)

    def apply_on_index(self, index: typing.Union[pd.Index, pd.DataFrame],
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

        .. _Unified Format: http://tools.pp.audeering.com/audata/data-format
            .html

        """  # noqa: E501
        if len(output_folder) == 0:
            raise ValueError('Please provide a valid string as the '
                             'output folder.')

        output_folder = audeer.safe_path(output_folder)
        mapping_file = os.path.join(output_folder, 'augmented.pkl')
        mapping_file_csv = mapping_file[:-4] + '.csv'
        transform_file = os.path.join(output_folder, 'augmented.yaml')

        if os.path.isfile(mapping_file) and os.path.isfile(transform_file):
            old_mapping = pd.read_pickle(mapping_file)
            old_transform = Transform.load(transform_file)
            if not old_mapping.index.equals(index) \
                    or old_transform != self.transform:
                warnings.warn(UserWarning((
                    'index or transform has changed, '
                    'consider setting force_overwrite=True')))
            if not force_overwrite:
                return old_mapping
            else:
                for file in old_mapping:
                    _remove(file)

        audeer.mkdir(output_folder)

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
        input_folder = audeer.safe_path(input_folder)
        output_folder = audeer.safe_path(output_folder)

        db = audata.Database.load(input_folder)

        audeer.mkdir(output_folder)
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


class Augment(audinterface.Process):
    r"""Augmentation interface.

    Args:
        transform: transformation object
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
        keep_nat: if the end of segment is set to ``NaT`` do not replace
            with file duration in the result
        num_workers: number of parallel jobs or 1 for sequential
            processing. If ``None`` will be set to the number of
            processors on the machine multiplied by 5 in case of
            multithreading and number of processors in case of
            multiprocessing
        multiprocessing: use multiprocessing instead of multithreading
        verbose: show debug messages

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    """
    def __init__(
            self,
            transform: Transform,
            *,
            sampling_rate: int = None,
            resample: bool = False,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
    ):
        self.transform = transform
        r"""The transformation object."""

        super().__init__(
            process_func=Augment._process_func,
            transform=transform,
            sampling_rate=sampling_rate,
            resample=resample,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

    def augment(
            self,
            column_or_table: typing.Union[pd.Series, pd.DataFrame],
            cache_root: str,
            *,
            channel: int = None,
            modified_only: bool = True,
            num_variants: int = 1,
            force: bool = False,
    ) -> typing.Union[pd.Series, pd.DataFrame]:
        r"""Apply data augmentation to a column or table in Unified Format.

        Creates ``num_variants`` copies of the files referenced in the index
        and augments every segment individually.
        Augmented files are stored as
        ``<cache_root>/<uid>/<variant>/<subdir>/<filename>``,
        where ``subdir`` is the directory tree that remains after removing
        the common directory shared by all files.
        Parts of the signals that are not covered by at least one segment
        are not augmented.
        The result is a column / table with a new index that
        references the augmented files, but the same column / table data.
        If ``num_variants > 1`` the data is duplicated accordingly.
        If ``modified_only`` is set to ``False`` includes the original
        index.

        Args:
            column_or_table: table
            cache_root: root directory under which augmented files are stored
            channel: channel number
            modified_only: return only modified segments, otherwise
                combine original and modified segments
            num_variants: number of variations that are created for every
                segment
            force: overwrite existing files

        Returns:
            column or table with new index

        Raises:
            RuntimeError: if sampling rates of file and transformation do not
                match

        """
        column_or_table = audata.utils.to_segmented_frame(column_or_table)
        modified = []

        if column_or_table.empty:
            return column_or_table

        for idx in range(num_variants):

            cache_root_idx = self._safe_cache(cache_root, idx)
            series = self._augment_index(
                column_or_table.index,
                cache_root_idx,
                channel,
                force,
            )

            if not modified_only and idx == 0:
                new_column_or_table = column_or_table.copy()
                new_column_or_table.index = series.index
                modified.append(new_column_or_table)

            new_level = [
                series[level].values[0] for level in series.index.levels[0]
            ]
            new_index = series.index.set_levels(new_level, level=0)
            new_column_or_table = column_or_table.copy()
            new_column_or_table.index = new_index
            modified.append(new_column_or_table)

        return pd.concat(modified, axis=0)

    def _augment_index(
            self,
            index: pd.MultiIndex,
            cache_root: str,
            channel: int,
            force: bool,
    ) -> pd.Series:
        r"""Augment from a segmented index and store augmented files to disk.
        Returns a series that points to the augmented files."""

        files = index.levels[0]
        out_files = Augment._out_files(files, cache_root)
        params = [
            (
                (
                    file,
                    out_file,
                    index[
                        index.get_level_values(0) == file
                    ].droplevel(0),
                    channel,
                    force),
                {},
            )
            for file, out_file in zip(files, out_files)
        ]
        segments = audeer.run_tasks(
            self._augment_file_to_cache,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=self.verbose,
            task_description=f'Process {len(index)} segments',
        )
        return pd.concat(segments)

    def _augment_file(
            self,
            file: str,
            index: pd.MultiIndex,
            channel: int,
    ) -> typing.Tuple[np.ndarray, int, pd.Series]:
        r"""Augment file at every segment in index.
        Returns the augmented signal, sampling rate and
        a series with the augmented segments."""

        signal, sampling_rate = self.read_audio(file, channel=channel)
        return self._augment_signal(signal, sampling_rate, index)

    def _augment_file_to_cache(
            self,
            file: str,
            out_file: str,
            index: pd.MultiIndex,
            channel: int,
            force: bool,
    ) -> pd.Series:
        r"""Augment 'file' and store result to 'out_file'.
        Return a series where every segment points to 'out_file'"""

        if force or not os.path.exists(out_file):
            signal, sampling_rate, segments = self._augment_file(
                file, index, channel,
            )
            index = segments.index
            audeer.mkdir(os.path.dirname(out_file))
            audiofile.write(out_file, signal, sampling_rate)
        else:
            index = self._correct_index(out_file, index)

        return pd.Series(
            out_file,
            index=pd.MultiIndex(
                levels=[[file], index.levels[0], index.levels[1]],
                codes=[[0] * len(index), index.codes[0], index.codes[1]],
                names=['file', 'start', 'end'],
            )
        )

    def _augment_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.MultiIndex,
    ) -> typing.Tuple[np.ndarray, int, pd.Series]:
        r"""Augment signal at every segment in index.
        Returns the augmented signal, sampling rate and
        a series with the augmented segments."""

        signal, sampling_rate = self._resample(signal, sampling_rate)
        new_index = super().process_signal_from_index(
            signal, sampling_rate, index,
        )
        for (start, end), segment in new_index.items():
            start_i = int(start.total_seconds() * sampling_rate)
            end_i = start_i + segment.shape[1]
            signal[:, start_i:end_i] = segment
        return signal, sampling_rate, new_index

    def _correct_index(
            self,
            file: str,
            index: pd.MultiIndex,
    ):
        r"""Replaces NaT in index."""

        # check if index contains NaT
        if -1 in index.codes[0] or -1 in index.codes[1]:

            # replace NaT in start with 0
            new_starts = index.get_level_values(0).map(
                lambda x: pd.to_timedelta(0) if pd.isna(x) else x
            )
            # if not keep_nat replace NaT in end with file duration
            if not self.keep_nat:
                dur = pd.to_timedelta(audiofile.duration(file), unit='s')
                new_ends = index.get_level_values(1).map(
                    lambda x: dur if pd.isna(x) else x
                )
            else:
                new_ends = index.get_level_values(1)

            index = pd.MultiIndex.from_arrays(
                [new_starts, new_ends], names=['start', 'end']
            )

        return index

    def _safe_cache(
            self,
            cache_root,
            idx: int,
    ) -> typing.Optional[str]:
        if cache_root is not None:
            cache_root = audeer.safe_path(cache_root)
            uid = audeer.uid(from_string=str(self.transform))
            cache_root = os.path.join(cache_root, uid, str(idx))
        return cache_root

    @staticmethod
    def _out_files(
            files: typing.Sequence[str],
            cache_root: str,
    ) -> typing.Sequence[str]:
        r"""Return cache file names by replacing the common directory path
        all files have in common with the cache directory."""
        files = [audeer.safe_path(file) for file in files]
        dirs = [os.path.dirname(file) for file in files]
        common_dir = audeer.common_directory(dirs)
        cache_root = audeer.safe_path(cache_root)
        out_files = [file.replace(common_dir, cache_root) for file in files]
        return out_files

    @staticmethod
    def _process_func(x, sr, transform):
        r"""Internal processing function: creates audio buffer
        and runs it through the transformation object."""
        with AudioBuffer.from_array(x.copy(), sr) as buf:
            transform(buf)
            return np.atleast_2d(buf.data.copy())
