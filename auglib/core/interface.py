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

from auglib.core.buffer import (
    AudioBuffer,
    Transform,
)


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


class NumpyTransform:
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


class Augment(audinterface.Process):
    r"""Augmentation interface.

    Provides an interface for :class:`auglib.Transform`
    and turns it into an object that can be applied on a list of files
    and data in the Unified Format.

    Note that all :meth:`auglib.Augment.process_*` methods return a column
    holding the augmented signals or segments,
    whereas :meth:`auglib.Augment.augment`
    stores the augmented signals back to disk
    and returns an index pointing to the augmented files.

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

    Example:
        >>> import audb
        >>> import audeer
        >>> import auglib
        >>> db = audb.load(
        ...     'testdata',
        ...     version='1.5.0',
        ...     verbose=False,
        ... )
        >>> transform = auglib.transform.WhiteNoiseUniform()
        >>> augmentation = auglib.Augment(transform)
        >>> column = db['emotion.test.gold']['emotion'].get()
        >>> augmented_column = augmentation.augment(
        ...     column,
        ...     'cache',
        ...     remove_root=db.meta['audb']['root'],
        ... )
        >>> label = augmented_column[0]  # file of first segment
        >>> file = augmented_column.index[0][0]  # label of first segment
        >>> file = file.replace(audeer.safe_path('.'), '.')  # remove absolute path
        >>> file, label
        ('./cache/37e84d07-b0c5-30dd-2e9e-c0e81e9fd127/0/audio/006.wav', 'unhappy')

    """  # noqa: E501
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
            data: typing.Union[pd.Index, pd.Series, pd.DataFrame],
            cache_root: str,
            *,
            remove_root: str = None,
            channel: int = None,
            modified_only: bool = True,
            num_variants: int = 1,
            force: bool = False,
    ) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
        r"""Augment an Unified Format index, column, or table.

        Creates ``num_variants`` copies of the files referenced in the index
        and augments them.
        If the index is segmented, only the segments are augmented.
        Augmented files are stored as
        ``<cache_root>/<uid>/<variant>/<original_path>``.
        It is possible to shorten the path by setting ``remove_root``
        to a directory that should be removed from ``<original_path>``
        (e.g. the ``audb`` cache folder).
        Parts of the signals that are not covered by at least one segment
        are not augmented.
        The result is an index, column or table that references the
        augmented files.
        If the input is a column or table data, the original content is kept.
        If ``num_variants > 1`` the data is duplicated accordingly.
        If ``modified_only`` is set to ``False`` the original index is also
        included.

        Args:
            data: index, column or table in Unified Format
            cache_root: root directory under which augmented files are stored
            remove_root: set a path that should be removed from
                the beginning of the original file path before joining with
                ``cache_root``
            channel: channel number starting from 0
            modified_only: return only modified segments, otherwise
                combine original and modified segments
            num_variants: number of variations that are created for every
                segment
            force: overwrite existing files

        Returns:
            index, column or table including augmented files

        Raises:
            RuntimeError: if sampling rates of file and transformation do not
                match

        """
        data = audata.utils.to_segmented_frame(data)
        modified = []

        if data.empty:
            return data

        if isinstance(data, pd.Index):
            index = data
            data = data.to_series()
            return_index = True
        else:
            index = data.index
            return_index = False

        for idx in range(num_variants):

            cache_root_idx = self._safe_cache(cache_root, idx)
            series = self._augment_index(
                index,
                cache_root_idx,
                remove_root,
                channel,
                force,
            )

            if not modified_only and idx == 0:
                new_data = data.copy()
                new_data.index = series.index
                modified.append(new_data)

            files = series.index.get_level_values(0).unique()
            new_level = [
                series[file].values[0] for file in files
            ]
            new_index = series.index.set_levels(new_level, level=0)
            new_data = data.copy()
            new_data.index = new_index
            modified.append(new_data)

        result = pd.concat(modified, axis=0)
        if return_index:
            return result.index
        else:
            return result

    def _augment_index(
            self,
            index: pd.MultiIndex,
            cache_root: str,
            remove_root: str,
            channel: int,
            force: bool,
    ) -> pd.Series:
        r"""Augment from a segmented index and store augmented files to disk.
        Returns a series that points to the augmented files."""

        files = index.get_level_values(0).unique()
        out_files = Augment._out_files(files, cache_root, remove_root)
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
            cache_root = os.path.realpath(audeer.safe_path(cache_root))
            uid = self.transform.id
            cache_root = os.path.join(cache_root, uid, str(idx))
        return cache_root

    @staticmethod
    def _out_files(
            files: typing.Sequence[str],
            cache_root: str,
            remove_root: str = None,
    ) -> typing.Sequence[str]:
        r"""Return cache file names by joining with the cache directory."""
        files = [audeer.safe_path(file) for file in files]
        cache_root = audeer.safe_path(cache_root)
        if remove_root is None:
            def join(path1: str, path2: str) -> str:
                seps = os.sep + os.altsep if os.altsep else os.sep
                return os.path.join(
                    path1, os.path.splitdrive(path2)[1].lstrip(seps),
                )
            out_files = [
                join(cache_root, file) for file in files
            ]
        else:
            remove_root = audeer.safe_path(remove_root)
            dirs = [os.path.dirname(file) for file in files]
            common_root = audeer.common_directory(dirs)
            if not audeer.common_directory(
                [remove_root, common_root]
            ) == remove_root:
                raise RuntimeError(f"Cannot remove '{remove_root}' "
                                   f"from '{common_root}'.")
            out_files = [
                file.replace(remove_root, cache_root, 1) for file in files
            ]
        return out_files

    @staticmethod
    def _process_func(x, sr, transform):
        r"""Internal processing function: creates audio buffer
        and runs it through the transformation object."""
        with AudioBuffer.from_array(x.copy(), sr) as buf:
            transform(buf)
            return np.atleast_2d(buf.data.copy())
