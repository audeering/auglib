import os
import typing

import numpy as np
import pandas as pd

import audeer
import audformat
import audinterface
from audinterface.core.utils import preprocess_signal
import audiofile
import audobject

import auglib
from auglib.core import transform
from auglib.core.seed import seed as seed_func


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
        >>> import audiofile
        >>> import auglib
        >>> db = audb.load(
        ...     "emodb",
        ...     version="1.4.1",
        ...     media=["wav/03a01Fa.wav", "wav/03a01Nc.wav", "wav/03a01Wa.wav"],
        ...     verbose=False,
        ... )
        >>> transform = auglib.transform.WhiteNoiseUniform()
        >>> augment = auglib.Augment(transform)
        >>> # Augment a numpy array
        >>> signal, sampling_rate = audiofile.read(db.files[0])
        >>> signal_augmented = augment(signal, sampling_rate)
        >>> # Augment (parts of) a database
        >>> df = db.get("emotion")
        >>> df_augmented = augment.augment(
        ...     df,
        ...     cache_root="cache",
        ...     remove_root=db.root,
        ... )
        >>> label = df_augmented.iloc[0, 0]
        >>> file = df_augmented.index[0][0]
        >>> file, label
        ('...03a01Fa.wav', 'happiness')

    """  # noqa: E501

    @audobject.init_decorator(
        hide=[
            "keep_nat",
            "multiprocessing",
            "num_workers",
            "verbose",
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
            process_func_args={"transform": transform},
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

        Creates ``num_variants`` copies of the segments referenced in the index
        and augments them.
        Files of augmented segments are stored as
        ``<cache_root>/<short_id>/<index_id>/<variant>/<original_path>``.
        The ``<index_id>`` is the identifier of the index of ``data``.
        Note that the ``<index_id>`` of a filewise index
        is the same as its corresponding segmented index
        with ``start=0`` and ``end=NaT``,
        but differs from its corresponding segmented index
        when ``end`` is set to the file durations.
        It is possible to shorten the path by setting ``remove_root``
        to a directory that should be removed from ``<original_path>``
        (e.g. the ``audb`` cache folder).
        If more than one segment of the same file is augmented,
        a counter is added at the end of the filename.
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

        # prepare index for hashing:
        # 1. convert to segmented index
        # 2. remember position of NaT
        # 3. convert to absolute file names

        is_index = isinstance(data, pd.Index)
        if is_index:
            index = data
        else:
            index = data.index
        original_index = index

        index = audformat.utils.to_segmented_index(
            index,
            allow_nat=True,
        )
        if self.keep_nat:
            nat_mask = index.get_level_values(audformat.define.IndexField.END).isna()

        if data_root is not None:
            index = audformat.utils.expand_file_path(index, data_root)

        index_hash = audformat.utils.hash(index)

        # figure out cache root

        if cache_root is None:
            # Import here to avoid circular import
            from auglib.core.cache import default_cache_root

            cache_root = default_cache_root()
        else:
            cache_root = audeer.path(cache_root, follow_symlink=True)
        cache_root = os.path.join(cache_root, self.short_id)

        transform_path = os.path.join(cache_root, "transform.yaml")
        if not os.path.exists(transform_path):
            self.transform.to_yaml(
                transform_path,
                include_version=False,
            )

        # apply augmentation

        # Holds index with non NaT timestamps if required
        non_nat_index = None

        if modified_only:
            augmented_indices = []
        else:
            if self.keep_nat:
                augmented_indices = [original_index]
            else:
                non_nat_index = audformat.utils.to_segmented_index(
                    index,
                    allow_nat=False,
                )
                augmented_indices = [non_nat_index]

        for idx in range(num_variants):
            cache_root_idx = os.path.join(cache_root, index_hash, str(idx))

            index_cache_path = os.path.join(
                cache_root_idx,
                "index.pkl",
            )
            if not force and os.path.exists(index_cache_path):
                augmented_index = pd.read_pickle(index_cache_path)
                # Make sure old cache entries use correct dtype
                augmented_index = audformat.utils.set_index_dtypes(
                    augmented_index,
                    {"file": "string"},
                )
            else:
                # We always replace NaT when storing in cache
                # as `keep_nat` is not serialized.
                # When `keep_nat` is `True`
                # we replace the index values later on with `_apply_nat_mask()`

                # Only compute the duration of segments once for all variants
                if non_nat_index is None:
                    non_nat_index = audformat.utils.to_segmented_index(
                        index,
                        allow_nat=False,
                    )

                augmented_index = self._augment_index(
                    non_nat_index,
                    cache_root_idx,
                    remove_root,
                    f"Augment ({idx+1} of {num_variants})",
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
        r"""Augment segments and store augmented files to cache.

        Args:
            index: segmented index
            cache_root: cache root for augmented files
            remove_root: directory that should be removed from
                the beginning of the original file path before joining with
                ``cache_root``
            description: text to display in progress bar

        Returns:
            segmented index of augmented files

        """
        files = index.get_level_values("file")
        starts = index.get_level_values("start")
        ends = index.get_level_values("end")
        augmented_files = _augmented_files(files, cache_root, remove_root)
        params = [
            ((file, start, end, out_file), {})
            for file, start, end, out_file in zip(files, starts, ends, augmented_files)
        ]
        durations = audeer.run_tasks(
            self._augment_file_to_cache,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=self.verbose,
            task_description=description,
        )
        augmented_index = audformat.segmented_index(
            augmented_files,
            [0] * len(durations),
            durations,
        )
        return augmented_index

    def _augment_file_to_cache(
        self,
        file: str,
        start: pd.Timedelta,
        end: pd.Timedelta,
        augmented_file: str,
    ) -> float:
        r"""Augment file and store to cache.

        Before augmenting the file,
        it is also resampled,
        or remixed,
        if required.

        Args:
            file: path to incoming audio file
            start: start time to read ``file``
            end: end time to read ``file``
            augmented_file: path of augmented file

        Returns:
            duration of augmented file in seconds

        """
        signal, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end
        )
        signal, sampling_rate = preprocess_signal(
            signal,
            sampling_rate=sampling_rate,
            expected_rate=self.sampling_rate,
            resample=self.resample,
            channels=self.channels,
            mixdown=self.mixdown,
        )
        augmented_signal = self(signal, sampling_rate)
        audeer.mkdir(os.path.dirname(augmented_file))
        audiofile.write(augmented_file, augmented_signal, sampling_rate)
        duration = augmented_signal.shape[1] / sampling_rate
        return duration

    @staticmethod
    def _process_func(
        signal: np.ndarray,
        sampling_rate: int,
        transform: auglib.transform.Base,
    ) -> np.ndarray:
        r"""Internal processing function.

        Creates audio signal
        and runs it through the transformation object.

        """
        signal = np.atleast_2d(signal)
        max_samples = 0
        signals = []

        # augment each channel individually
        for channel in signal:
            channel = transform(channel, sampling_rate)
            max_samples = max(max_samples, channel.shape[0])
            signals.append(channel)

        # combine into single output array and fill short channels with zeros
        augmented_signal = np.zeros(
            (signal.shape[0], max_samples),
            dtype="float32",
        )
        for idx, channel in enumerate(signals):
            augmented_signal[idx, : len(channel)] = channel

        return augmented_signal


def _apply_nat_mask(
    index: pd.Index,
    mask: np.ndarray,
) -> pd.Index:
    r"""Within mask set end level to NaT."""
    ends = index.get_level_values(audformat.define.IndexField.END).to_series()
    ends[mask] = pd.NaT
    index = audformat.segmented_index(
        index.get_level_values(audformat.define.IndexField.FILE),
        index.get_level_values(audformat.define.IndexField.START),
        ends,
    )

    return index


def _augmented_files(
    files: typing.Sequence[str],
    cache_root: str,
    remove_root: str = None,
) -> typing.List[str]:
    r"""Return destination path for augmented files.

    If files contain the same filename several times,
    e.g. when augmenting segments,
    it will convert them into unique filenames:

        <augmented_file>-0
        <augmented_file>-1
        ...

    As segments are stored as single files.

    If we have more than 10 files,
    the counter will use two digits:

        <augmented_file>-00
        <augmented_file>-01
        ...

    and so on.

    Args:
        files: files to augment
        cache_root: cache root of augmented files
        remove_root: directory that should be removed from
            the beginning of the original file path before joining with
            ``cache_root``

    Returns:
        path of augmented files

    """
    # Estimate number of segments/samples for each file
    unique_files, counts = np.unique(files, return_counts=True)
    counts = {file: count for file, count in zip(unique_files, counts)}
    current_count = {file: 0 for file in unique_files}

    augmented_files = []
    for file in files:
        if counts[file] > 1:
            digits = len(str(counts[file] - 1))
            root, ext = os.path.splitext(file)
            augmented_file = f"{root}-{str(current_count[file]).zfill(digits)}{ext}"
        else:
            augmented_file = file
        current_count[file] += 1
        augmented_files.append(augmented_file)

    if remove_root is None:

        def join(path1: str, path2: str) -> str:
            seps = os.sep + os.altsep if os.altsep else os.sep
            return os.path.join(
                path1,
                os.path.splitdrive(path2)[1].lstrip(seps),
            )

        augmented_files = [join(cache_root, file) for file in augmented_files]
    else:
        remove_root = audeer.path(remove_root)
        dirs = [os.path.dirname(file) for file in unique_files]
        common_root = audeer.common_directory(dirs)
        if not audeer.common_directory([remove_root, common_root]) == remove_root:
            raise RuntimeError(
                f"Cannot remove '{remove_root}' " f"from '{common_root}'."
            )
        augmented_files = [
            file.replace(remove_root, cache_root, 1) for file in augmented_files
        ]
    return augmented_files
