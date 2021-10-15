import os
import shutil
import typing

import numpy as np
import pandas as pd

import audeer
import audformat
import audinterface
from audinterface.core.utils import preprocess_signal
import audiofile
import audobject

from auglib.core import transform
from auglib.core.buffer import AudioBuffer
from auglib.core.config import config


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


@audeer.deprecated(removal_version='1.0.0', alternative='auglib.Augment')
class NumpyTransform:
    r"""Interface for data augmentation on numpy arrays.

    Wraps a :class:`auglib.Transform` under a ``Callable``
    interface, allowing it to be called on-the-fly on numpy arrays.

    Args:
        transform: auglib transform to be wrapped
        sampling_rate: sampling rate of incoming signals

    """

    def __init__(self, transform: transform.Base, sampling_rate: int):
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        with AudioBuffer.from_array(signal, self.sampling_rate) as buf:
            self.transform(buf)
            transformed = buf.to_array()
        return transformed


class Augment(audinterface.Process, audobject.Object):
    r"""Augmentation interface.

    Provides an interface for :class:`auglib.Transform`
    and turns it into an object
    that can be applied
    on a signal,
    file(s)
    and an audformat_ database.
    More details are discussed under :ref:`usage`.

    .. code-block:: python

        augment = auglib.Augment(transform)
        # Apply on signal, returns np.ndarray
        signal = augment(signal, sampling_rate)
        # Apply on signal, file, files, database index.
        # Returns a column holding the augmented signals
        y = augment.process_signal(signal, sampling_rate)
        y = augment.process_file(file)
        y = augment.process_files(files)
        y = augment.process_index(index)
        # Apply on index, table, column.
        # Writes results to disk
        # and returns index, table, column
        # pointing to augmented files
        index = augment.augment(index)
        y = augment.augment(y)
        df = augment.augment(df)

    :class:`auglib.Augment` inherits from :class:`audobject.Object`,
    which means you can seralize to
    and instantiate the class
    from a YAML file.
    Have a look at :class:`audobject.Object`
    to see all available methods.
    The following arguments are not serialized:
    ``keep_nat``,
    ``multiprocessing``,
    ``num_workers``,
    ``verbose``.
    For more information see section on `hidden arguments`_.

    .. _audformat: https://audeering.github.io/audformat/data-format.html

    Args:
        transform: transformation object
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
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

    .. _`hidden arguments`: https://audeering.github.io/audobject/usage.html#hidden-arguments

    Example:
        >>> import audb
        >>> import audeer
        >>> import audiofile
        >>> import auglib
        >>> db = audb.load(
        ...     'testdata',
        ...     version='1.5.0',
        ...     verbose=False,
        ... )
        >>> transform = auglib.transform.WhiteNoiseUniform()
        >>> augment = auglib.Augment(transform)
        >>> # Augment a numpy array
        >>> signal, sampling_rate = audiofile.read(db.files[0])
        >>> signal_augmented = augment(signal, sampling_rate)
        >>> # Augment (parts of) a database
        >>> column = db['emotion.test.gold']['emotion'].get()
        >>> augmented_column = augment.augment(
        ...     column,
        ...     cache_root='cache',
        ...     remove_root=db.meta['audb']['root'],
        ... )
        >>> label = augmented_column[0]
        >>> file = augmented_column.index[0][0]
        >>> file = file.replace(audeer.safe_path('.'), '.')  # remove absolute path
        >>> file, label
        ('./cache/0482e0f5-3013-2222-dad7-481251925619/0/audio/006.wav', 'unhappy')

    """  # noqa: E501
    @audobject.init_decorator(
        hide=[
            'keep_nat',
            'multiprocessing',
            'num_workers',
            'verbose',
        ]
    )
    def __init__(
            self,
            transform: transform.Base,
            *,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = None,
            mixdown: bool = False,
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
            channels=channels,
            mixdown=mixdown,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

    @audeer.deprecated_keyword_argument(
        deprecated_argument='channel',
        removal_version='0.10.0',
    )
    def augment(
            self,
            data: typing.Union[pd.Index, pd.Series, pd.DataFrame],
            cache_root: str = None,
            *,
            remove_root: str = None,
            modified_only: bool = True,
            num_variants: int = 1,
            force: bool = False,
    ) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
        r"""Augment an index, column, or table conform to audformat.

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
            data: index, column or table conform to audformat
            cache_root: directory to cache augmented files,
                if ``None`` defaults to ``auglib.config.CACHE_ROOT``
            remove_root: set a path that should be removed from
                the beginning of the original file path before joining with
                ``cache_root``
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
            RuntimeError: if ``modified_only=False`` but resampling or
                remixing is turned on

        """
        if not modified_only:
            if self.channels is not None or self.mixdown or self.resample:
                raise ValueError(
                    "It is not possible to use 'modified_only=False' "
                    "if remixing or resampling is turned on. "
                    "To avoid this error, set arguments "
                    "'channels', 'mixdown' and 'resample' "
                    "to their default values."
                )

        data = audformat.utils.to_segmented_index(data)
        if data.empty:
            return data

        modified = []
        if cache_root is None:
            cache_root = default_cache_root(self)
        else:
            cache_root = audeer.safe_path(
                os.path.join(cache_root, self.id)
            )

        # save yaml of transform to cache
        self.transform.to_yaml(
            os.path.join(cache_root, 'transform.yaml'),
            include_version=False,
        )

        if isinstance(data, pd.Index):
            index = data
            data = data.to_series()
            return_index = True
        else:
            index = data.index
            return_index = False

        for idx in range(num_variants):

            series = self._augment_index(
                index,
                os.path.join(cache_root, str(idx)),
                remove_root,
                force,
                f'Augment ({idx+1} of {num_variants})',
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
            force: bool,
            description: str,
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
                    force),
                {},
            )
            for file, out_file in zip(files, out_files)
        ]
        verbose = self.verbose
        self.verbose = False  # avoid nested progress bar
        segments = audeer.run_tasks(
            self._augment_file_to_cache,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=verbose,
            task_description=description,
        )
        self.verbose = verbose
        return pd.concat(segments)

    def _augment_file(
            self,
            file: str,
            index: pd.MultiIndex,
    ) -> typing.Tuple[np.ndarray, int, pd.Series]:
        r"""Augment file at every segment in index.
        Returns the augmented signal, sampling rate and
        a series with the augmented segments."""

        signal, sampling_rate = audiofile.read(file, always_2d=True)
        return self._augment_signal(signal, sampling_rate, index)

    def _augment_file_to_cache(
            self,
            file: str,
            out_file: str,
            index: pd.MultiIndex,
            force: bool,
    ) -> pd.Series:
        r"""Augment 'file' and store result to 'out_file'.
        Return a series where every segment points to 'out_file'"""

        if force or not os.path.exists(out_file):
            signal, sampling_rate, segments = self._augment_file(file, index)
            index = segments.index
            audeer.mkdir(os.path.dirname(out_file))
            audiofile.write(out_file, signal, sampling_rate)
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

        signal, sampling_rate = preprocess_signal(
            signal,
            sampling_rate,
            expected_rate=self.sampling_rate,
            resample=self.resample,
            channels=self.channels,
            mixdown=self.mixdown,
        )
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
                new_ends = index.get_level_values(1).map(
                    lambda x: pd.to_timedelta(
                        audiofile.duration(file),
                        unit='s',
                    )
                    if pd.isna(x)
                    else pd.to_timedelta(x)
                )
            else:
                new_ends = index.get_level_values(1).map(
                    lambda x: pd.to_timedelta(x, unit='s')
                )
            index = audinterface.utils.signal_index(
                starts=new_starts,
                ends=new_ends,
            )

        return index

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
            return np.atleast_2d(buf._data.copy())


def clear_default_cache_root(augment: Augment = None):
    r"""Clear default cache directory.

    If ``augment`` is not None,
    deletes only the sub-directory where files
    created by the :class:`auglib.Augment` object are stored.

    Args:
        augment: optional augmentation object

    """
    root = default_cache_root(augment)
    if os.path.exists(root):
        shutil.rmtree(root)
    if augment is None:
        audeer.mkdir(root)


def default_cache_root(augment: Augment = None) -> str:
    r"""Path to default cache directory.

    The default cache directory defines
    the path where augmented files will be stored.
    It is given by the path specified
    by the environment variable
    ``AUGLIB_CACHE_ROOT``
    or by
    ``auglib.config.CACHE_ROOT``.

    If ``augment`` is not None,
    returns the sub-directory where files
    created by the :class:`auglib.Augment` object are stored.

    Args:
        augment: optional augmentation object

    Returns:
        cache directory path

    """
    root = (
        os.environ.get('AUGLIB_CACHE_ROOT')
        or config.CACHE_ROOT
    )
    if augment is not None:
        root = os.path.join(root, augment.id)
    return audeer.safe_path(root)
