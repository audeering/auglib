import os
import typing

import numpy as np
import pandas as pd

import audeer
import audformat
import audinterface
from audinterface.core.utils import preprocess_signal
import audiofile
import audmath
import audobject

import auglib
from auglib.core import transform
from auglib.core.buffer import AudioBuffer
from auglib.core.seed import seed as seed_func


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
class NumpyTransform:  # pragma: no cover
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
    If input has multiple channels,
    each channel is augmented individually.
    I.e. in case randomized arguments are used,
    the augmentation can be different for each channel.
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
    which means you can serialize to
    and instantiate the class
    from a YAML file.
    By setting a ``seed``,
    you can further ensure
    that re-running the augmentation loaded form a YAML file
    will create the same output.
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
            multiprocessing.
            If ``seed`` is not ``None``,
            the value is always set to 1
        multiprocessing: use multiprocessing instead of multithreading
        seed: if not ``None`` calls :func:`auglib.seed` with the
            given value when object is constructed.
            This will automatically set ``num_workers`` to 1
        verbose: show debug messages

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    .. _`hidden arguments`: https://audeering.github.io/audobject/usage.html#hidden-arguments

    Examples:
        >>> import audb
        >>> import audeer
        >>> import audiofile
        >>> import auglib
        >>> db = audb.load(
        ...     'testdata',
        ...     version='1.5.0',
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> transform = auglib.transform.WhiteNoiseUniform()
        >>> augment = auglib.Augment(transform)
        >>> # Augment a numpy array
        >>> file = os.path.join(db.root, db.files[0])
        >>> signal, sampling_rate = audiofile.read(file)
        >>> signal_augmented = augment(signal, sampling_rate)
        >>> # Augment (parts of) a database
        >>> column = db['emotion.test.gold']['emotion'].get()
        >>> augmented_column = augment.augment(
        ...     column,
        ...     cache_root='cache',
        ...     data_root=db.root,
        ... )
        >>> label = augmented_column[0]
        >>> file = augmented_column.index[0][0]
        >>> file = file.replace(audeer.safe_path('.'), '.')  # remove absolute path
        >>> file, label  # doctest: +SKIP
        ('./cache/decdec83/-6100627981361312836/0/audio/006.wav', 'unhappy')

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
            seed: int = None,
            verbose: bool = False,
    ):
        if seed is not None:
            seed_func(seed)
            num_workers = 1

        self.seed = seed
        r"""Random seed to initialize the random number generator."""
        self.transform = transform
        r"""The transformation object."""

        super().__init__(
            process_func=Augment._process_func,
            process_func_args={'transform': transform},
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

    @property
    def short_id(
            self,
    ) -> str:
        r"""Short flavor ID.

        This just truncates the ID
        to its last eight characters.

        """
        return self.id[-8:]

    @audeer.deprecated_keyword_argument(
        deprecated_argument='channel',
        removal_version='0.10.0',
    )
    def augment(
            self,
            data: typing.Union[pd.Index, pd.Series, pd.DataFrame],
            cache_root: str = None,
            *,
            data_root: str = None,
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
            data_root: if index contains relative files,
                set to root directory where files are stored
            remove_root: directory that should be removed from
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

        if len(data) == 0:
            return data

        # prepare index:
        # 1. remember position of NaT
        # 2. convert to absolute file names
        # 3. convert to segment index and replace NaT with duration

        is_index = isinstance(data, pd.Index)
        if is_index:
            index = data
        else:
            index = data.index
        original_index = index

        if self.keep_nat:
            index = audformat.utils.to_segmented_index(
                index,
                allow_nat=True,
            )
            nat_mask = index.get_level_values(
                audformat.define.IndexField.END
            ).isna()

        if data_root is not None:
            index = audformat.utils.expand_file_path(index, data_root)

        index = audformat.utils.to_segmented_index(
            index,
            allow_nat=False,
        )

        index_hash = audformat.utils.hash(index)

        # figure out cache root

        if cache_root is None:
            # Import here to avoid circular import
            from auglib.core.cache import default_cache_root
            cache_root = default_cache_root()
        else:
            cache_root = audeer.safe_path(cache_root)
        cache_root = os.path.join(cache_root, self.short_id)

        transform_path = os.path.join(cache_root, 'transform.yaml')
        if not os.path.exists(transform_path):
            self.transform.to_yaml(
                transform_path,
                include_version=False,
            )

        # apply augmentation

        if modified_only:
            augmented_indices = []
        else:
            if self.keep_nat:
                augmented_indices = [original_index]
            else:
                augmented_indices = [index]

        for idx in range(num_variants):

            cache_root_idx = os.path.join(
                cache_root,
                index_hash,
                str(idx)
            )

            index_cache_path = os.path.join(
                cache_root_idx,
                'index.pkl',
            )
            if not force and os.path.exists(index_cache_path):
                augmented_index = pd.read_pickle(index_cache_path)
                # Make sure old cache entries use correct dtype
                augmented_index = audformat.utils.set_index_dtypes(
                    augmented_index,
                    {'file': 'string'},
                )
            else:
                augmented_index = self._augment_index(
                    index,
                    cache_root_idx,
                    remove_root,
                    f'Augment ({idx+1} of {num_variants})',
                )
                pd.to_pickle(
                    augmented_index,
                    index_cache_path,
                )
            if self.keep_nat:
                augmented_index = _apply_nat_mask(
                    augmented_index,
                    nat_mask,
                )
            augmented_indices.append(augmented_index)

        # create augmented data

        if is_index:
            augmented_data = audformat.utils.union(augmented_indices)
        else:
            augmented_data = []
            for augmented_index in augmented_indices:
                augmented_data.append(data.set_axis(augmented_index))
            augmented_data = audformat.utils.concat(augmented_data)

        return augmented_data

    def _augment_index(
            self,
            index: pd.Index,
            cache_root: str,
            remove_root: str,
            description: str,
    ) -> pd.Index:
        r"""Augment segments and store augmented files to cache."""
        files = index.get_level_values(0).unique()
        augmented_files = _augmented_files(
            files,
            cache_root,
            remove_root,
        )
        params = [
            (
                (
                    file,
                    out_file,
                    index[
                        index.get_level_values(0) == file
                    ].droplevel(0),
                ),
                {},
            )
            for file, out_file in zip(files, augmented_files)
        ]

        verbose = self.verbose
        self.verbose = False  # avoid nested progress bar
        augmented_indices = audeer.run_tasks(
            self._augment_file_to_cache,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=verbose,
            task_description=description,
        )
        self.verbose = verbose

        augmented_index = audformat.utils.union(augmented_indices)

        return augmented_index

    def _augment_file(
            self,
            file: str,
            index: pd.Index,
    ) -> typing.Tuple[np.ndarray, int, pd.Index]:
        r"""Augment file at every segment."""
        signal, sampling_rate = audiofile.read(file, always_2d=True)
        augmented_signal, augmented_rate, augmented_index = \
            self._augment_signal(signal, sampling_rate, index)

        return augmented_signal, augmented_rate, augmented_index

    def _augment_file_to_cache(
            self,
            file: str,
            augmented_file: str,
            index: pd.Index,
    ) -> pd.Index:
        r"""Augment file and store to cache."""
        signal, sampling_rate, augmented_index = self._augment_file(
            file,
            index,
        )
        audeer.mkdir(os.path.dirname(augmented_file))
        audiofile.write(augmented_file, signal, sampling_rate)

        # insert augmented file name at first level
        augmented_index_with_file = pd.MultiIndex(
            levels=[
                [augmented_file],
                augmented_index.levels[0],
                augmented_index.levels[1]
            ],
            codes=[
                [0] * len(index),
                augmented_index.codes[0],
                augmented_index.codes[1],
            ],
            names=[
                audformat.define.IndexField.FILE,
                audformat.define.IndexField.START,
                audformat.define.IndexField.END
            ],
        )
        augmented_index_with_file = audformat.utils.set_index_dtypes(
            augmented_index_with_file,
            {'file': 'string'},
        )

        return augmented_index_with_file

    def _augment_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.Index,
    ) -> typing.Tuple[np.ndarray, int, pd.Index]:
        r"""Augment signal at every segment in index."""
        signal, augmented_rate = preprocess_signal(
            signal,
            sampling_rate=sampling_rate,
            expected_rate=self.sampling_rate,
            resample=self.resample,
            channels=self.channels,
            mixdown=self.mixdown,
        )
        y_augmented = self.process_signal_from_index(
            signal,
            augmented_rate,
            index,
        )
        # fix index in case augmentation has changed the segments
        y_augmented = _fix_segments(y_augmented, augmented_rate)

        # if augmented segments match original segments
        # we can replace processed segments in original signal
        # otherwise, we have to create the augmented signal
        # by concatenating unprocessed and augment segments
        if y_augmented.index.equals(index):
            _insert_segments_into_signal(y_augmented, signal, augmented_rate)
            augmented_signal = signal
        else:

            # series with unprocessed segments
            dur = pd.to_timedelta(
                signal.shape[1] / augmented_rate,
                unit='s',
            )
            y_unprocessed = audinterface.Process(
                process_func=lambda x, sr: x,
            ).process_signal_from_index(
                signal,
                augmented_rate,
                _invert_index(index, dur),
            )

            if y_unprocessed.empty:

                # calculate new signal duration
                new_dur = _index_duration(y_augmented.index).sum()

                # create empty signal and insert processed segments
                num_samples = int(
                    round(new_dur.total_seconds() * augmented_rate))
                augmented_signal = np.zeros((1, num_samples))
                _insert_segments_into_signal(
                    y_augmented,
                    augmented_signal,
                    augmented_rate,
                )

            else:

                # calculate new signal duration
                # and fix index of unprocessed segments
                new_dur = _index_duration(y_augmented.index).sum() +\
                    _index_duration(y_unprocessed.index).sum()
                y_unprocessed.index = _invert_index(
                    y_augmented.index,
                    new_dur,
                )

                # create empty signal and insert
                # processed and unprocessed segments
                num_samples = int(
                    round(new_dur.total_seconds() * augmented_rate)
                )
                augmented_signal = np.zeros((1, num_samples))
                _insert_segments_into_signal(
                    y_augmented,
                    augmented_signal,
                    augmented_rate,
                )
                _insert_segments_into_signal(
                    y_unprocessed,
                    augmented_signal,
                    augmented_rate,
                )

        return augmented_signal, augmented_rate, y_augmented.index

    @staticmethod
    def _process_func(
            signal: np.ndarray,
            sampling_rate: int,
            transform: auglib.transform.Base,
    ) -> np.ndarray:
        r"""Internal processing function.

        Creates audio buffer
        and runs it through the transformation object.

        """
        signal = np.atleast_2d(signal)
        max_samples = 0
        buffers = []

        # augment each channel individually
        for channel in signal:
            buffer = AudioBuffer.from_array(channel.copy(), sampling_rate)
            transform(buffer)
            max_samples = max(max_samples, len(buffer))
            buffers.append(buffer)

        # combine into single output array and fill short channels with zeros
        augmented_signal = np.zeros(
            (signal.shape[0], max_samples),
            dtype=np.float32,
        )
        for idx, buffer in enumerate(buffers):
            augmented_signal[idx, :len(buffer)] = buffer.to_array(copy=False)
            buffer.free()

        return augmented_signal


def _apply_nat_mask(
        index: pd.Index,
        mask: np.ndarray,
) -> pd.Index:
    r"""Within mask set end level to NaT."""
    ends = index.get_level_values(
        audformat.define.IndexField.END
    ).to_series()
    ends[mask] = pd.NaT
    index = audformat.segmented_index(
        index.get_level_values(
            audformat.define.IndexField.FILE
        ),
        index.get_level_values(
            audformat.define.IndexField.START
        ),
        ends,
    )

    return index


def _augmented_files(
        files: typing.Sequence[str],
        cache_root: str,
        remove_root: str = None,
) -> typing.Sequence[str]:
    r"""Return cache file names by joining with the cache directory."""
    cache_root = audeer.safe_path(cache_root)
    if remove_root is None:
        def join(path1: str, path2: str) -> str:
            seps = os.sep + os.altsep if os.altsep else os.sep
            return os.path.join(
                path1, os.path.splitdrive(path2)[1].lstrip(seps),
            )
        augmented_files = [
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
        augmented_files = [
            file.replace(remove_root, cache_root, 1) for file in files
        ]
    return augmented_files


def _fix_segments(
        y: pd.Series,
        sampling_rate: int,
) -> pd.Series:
    r"""Align index with signal samples."""
    delta = pd.to_timedelta(0)
    new_starts = []
    new_ends = []

    for (start, end), signal in y.items():
        start_i = audmath.samples(start.total_seconds(), sampling_rate)
        end_i = audmath.samples(end.total_seconds(), sampling_rate)
        duration_i = end_i - start_i
        new_starts.append(start + delta)
        delta += pd.to_timedelta(
            (signal.shape[1] - duration_i) / sampling_rate,
            unit='s',
        )
        new_ends.append(end + delta)

    y.index = audinterface.utils.signal_index(new_starts, new_ends)

    return y


def _index_duration(
        index: pd.Index,
) -> pd.TimedeltaIndex:
    r"""Calculate segment duration."""
    starts = index.get_level_values(audformat.define.IndexField.START)
    ends = index.get_level_values(audformat.define.IndexField.END)

    return ends - starts


def _invert_index(
        index: pd.Index,
        duration: pd.Timedelta,
) -> pd.Index:
    r"""Invert an index with start and end timestamps."""
    ends = index.get_level_values(audformat.define.IndexField.START)
    starts = index.get_level_values(audformat.define.IndexField.END)

    # if original index starts at 0
    # skip first entry in ends
    # otherwise prepend 0 to starts
    if ends[0] == pd.to_timedelta(0):
        ends = ends[1:]
    else:
        starts = starts.insert(0, pd.to_timedelta(0))

    # if original index ends at duration
    # skip last entry in starts
    # otherwise append duration to ends
    if starts[-1] == duration:
        starts = starts[:-1]
    else:
        ends = ends.insert(len(ends), duration)

    return audinterface.utils.signal_index(starts, ends)


def _insert_segments_into_signal(
        y: pd.Series,
        signal: np.ndarray,
        sampling_rate: int,
):
    r"""Insert segments into signal."""
    for (start, end), segment in y.items():
        start_i = audmath.samples(start.total_seconds(), sampling_rate)
        end_i = start_i + segment.shape[1]
        signal[:, start_i:end_i] = segment
