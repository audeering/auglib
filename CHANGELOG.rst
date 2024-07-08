Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.0.4 (2024-07-08)
--------------------------

* Fixed: preserve order of segments in index
  after augmentation with ``auglib.Augment.augment()``
* Fixed: ensure transforms work with signals
  stored in a read-only ``numpy.array``
* Fixed: return correct float types with ``numpy>=2.0``


Version 1.0.3 (2024-05-14)
--------------------------

* Fixed: broken links
  on the landing page
  of the documentation


Version 1.0.2 (2024-01-30)
--------------------------

* Fixed: ``auglib.transform.PinkNoise``
  when applied to signals with odd sample length


Version 1.0.1 (2024-01-26)
--------------------------

* Fixed: missing audio examples in documentation
* Fixed: add missing entries to the changelog of 1.0.0


Version 1.0.0 (2024-01-26)
--------------------------

* Added: public release
* Added: support for MacOS
* Added: support for Windows
* Added: audio examples and figures
  to the docstrings
  of all transforms
* Added: depend on ``scipy``
* Added: ``sampling_rate`` as argument
  to ``__call__()`` of transforms
  that require a sampling rate
* Added: support for Python 3.9, 3.10 and 3.11
* Added: ``auglib.utils.rms_db()``
* Changed: ``auglib.Augment.augment()``
  now stores augmented segments
  as single files in cache
* Changed: ``auglib.Augment.augment()``
  now uses ``allow_nat=True``
  when calculating the hash of an index,
  which is used to store/find augmented files
  in cache
* Changed: depend on ``audeer>=2.0.0``
* Changed: convert code base to pure Python
* Changed: switch to MIT License
* Fixed: support overlapping segments
* Fixed: ensure the ``duration`` argument
  of ``auglib.transform.AppendValue``
  and ``auglib.transform.PrependValue``
  is first observed
  when calling the transform
* Removed: ``sounddevice`` dependency
* Removed: ``auglib.AudioBuffer``
* Removed: ``auglib.set_exception_handling()``


Version 0.12.2 (2023-07-31)
---------------------------

* Fixed: changes to the correction
  of ``start`` and ``end``
  in ``auglib.Augment.augment()``
  as introduced in ``auglib`` v0.12.1


Version 0.12.1 (2023-07-27)
---------------------------

* Changed: require ``audmath >=1.3.0``
* Changed: require ``audinterface >=1.0.4``
* Changed: use ``audmath.samples()``
  to convert from duration to samples
* Fixed: add missing "Returns" section
  to documentation of ``auglib.AudioBuffer.from_array()``


Version 0.12.0 (2023-04-12)
---------------------------

* Added: ``auglib.transform.Fade``
* Changed: use version 2.0.3 of the ``auglib`` C++ library
* Changed: split API documentation into sub-pages
  for each function/class
* Changed: if ``loop_aux`` is ``True``
  in ``auglib.transform.Mix``
  the ``aux`` buffer is now first looped
  before ``read_pos_aux``
  and ``read_dur_aux``
  are applied
* Fixed: ``gain_base_db``
  in ``auglib.transform.Mix``
  was not applied
  to all samples of the ``base`` buffer
  starting at the length of the ``aux`` buffer
  and ending at ``read_dur_aux``


Version 0.11.2 (2022-12-08)
---------------------------

* Changed: require ``audobject >=0.7.6``
* Fixed: missing ``unit`` attribute
  in ``auglib.transform.BabbleNoise``


Version 0.11.1 (2022-10-17)
---------------------------

* Fixed: ``preserve_level`` argument was missing for
  ``auglib.transform.BabbleNoise``
  and ``auglib.transform.PrependValue``


Version 0.11.0 (2022-10-13)
---------------------------

* Added: ``auglib.transform.BabbleNoise``
* Added: ``auglib.transform.Prepend``
* Added: ``auglib.transform.PrependValue``
* Added: ``auglib.transform.Resample``
* Added: ``auglib.transform.Shift``
* Added: ``snr_db`` argument to
  ``auglib.transform.Mix``
  ``auglib.transform.PinkNoise``,
  ``auglib.transform.Tone``,
  ``auglib.transform.WhiteNoiseGaussian``,
  ``auglib.transform.WhiteNoiseUniform``
  to specify the signal-to-noise ratio
  between the incoming
  and added signal
* Added: ``end_pos`` and ``fill_pos`` to
  ``auglib.transform.Trim``
  to allow any kind of cropping,
  zero padding,
  and signal repetition
* Added: ``preserve_level`` argument
  to all transforms.
  If ``True`` it ensures
  that the transformed signal
  has the same root-mean-square value
  as the incoming signal
* Added: ``auglib.AudioBuffer.rms()``
  and ``auglib.AudioBuffer.rms_db()``
* Added: ``sampling_rate`` argument to
  ``auglib.AudioBuffer.write()``
* Added: *Noise with fixed SNR* example
  in the documentation
* Added: *Band-Pass Filtered Noise* example
  in the documentation
* Changed: allow a transform object as input
  for the ``aux`` argument
  in transforms that have this argument
* Changed: ``auglib.transform.Trim``
  no longer supports providing a ``start_pos``
  that is larger than the buffer length
* Changed: ``auglib.transform.Trim``
  with argument ``fill="loop"``
  no longer loops the whole input signal,
  but only the trimmed version
  specified by ``start_pos``
  and/or ``end_pos``.
  For cycling through a signal
  use ``auglib.transform.Shift`` instead
* Changed: raise ``ValueError``
  if ``sampling_rate`` argument
  is not greater than zero or not an integer
* Changed: serializing a transform
  that contains a buffer as ``aux`` argument
  will raise a ``ValueError``
* Changed: when applied to an input signal
  ``auglib.transform.Function``
  raises a ``RuntimeError``
  if it would result in an empty augmented signal
* Changed: depend on ``audformat>=0.15.2``
* Fixed: ``read_dur_aux``
  in ``auglib.transform.Append``
  does now allow to be ``None``


Version 0.10.5 (2022-06-15)
---------------------------

* Added: pitch shift example
* Added: constant pitch examples
* Fixed: correct examples for ``pedalboard>=0.4.0``


Version 0.10.4 (2022-01-26)
---------------------------

* Changed: depend on ``audformat>=0.13.0``
* Fixed: do not sort augmented index


Version 0.10.3 (2022-01-21)
---------------------------

* Fixed: Ci job for Python package publication


Version 0.10.2 (2022-01-21)
---------------------------

* Changed: use ``audinterface>=0.8.0``


Version 0.10.1 (2021-12-30)
---------------------------

* Changed: use Python 3.8 as default Python version


Version 0.10.0 (2021-11-17)
---------------------------

* Changed: ``Augment.augment()`` caches augmented index
* Fixed: ``Augment.augment()`` supports transforms that change the segment length
* Fixed: ``Augment.augment()`` supports index with relative file names


Version 0.9.0 (2021-10-25)
--------------------------

* Added: ``auglib.transform.Function``
* Added: Examples section to Getting Started part of the documentation
* Added: External Solutions section
  to Getting Started part of the documentation
* Added: ``auglib.observe.Base``
* Added: ``auglib.observe.Bool``
* Added: ``auglib.observe.FloatNorm``
* Added: ``auglib.observe.FloatUni``
* Added: ``auglib.observe.IntUni``
* Added: ``auglib.observe.List``
* Added: ``auglib.observe.observe``
* Added: ``num_repeat`` argument to ``auglib.transform.Mix``
* Added: ``auglib.seed()``
* Added: ``auglib.transform.Base``
* Added: ``auglib.Time``
* Added: ``fill`` argument to ``auglib.transform.Trim``
* Added: ``auglib.transform.Mask``
* Added: ``seed`` argument to ``auglib.Augment``
* Added: documentation on how to implement a transform
  under ``auglib.transform``
* Added: documentation on how to implement an observable
  under ``auglib.observe``
* Added: ``auglib.AudioBuffer.duration``
* Added: multi-channel support for ``auglib.Augment``
* Changed: ``auglib.AudioBuffer.to_array()`` returns 2d array
* Changed: hide ``AudioBuffer.data`` and ``AudioBuffer.obj``
* Changed: make ``sampling_rate`` a keyword argument
  in ``auglib.utils.to_samples()``
* Changed: increase code coverage to 100%
* Changed: use short ID for flavor folders in cache
* Deprecated: ``auglib.Int``
* Deprecated: ``auglib.IntList``
* Deprecated: ``auglib.Float``
* Deprecated: ``auglib.FloatList``
* Deprecated: ``auglib.Number``
* Deprecated: ``auglib.Str``
* Deprecated: ``auglib.StrList``
* Deprecated: ``auglib.NumpyTransform``
* Deprecated: ``auglib.utils.random_seed()``
* Deprecated: ``auglib.Transform``
* Deprecated: ``auglib.Source``
* Deprecated: ``auglib.source.FromArray``
* Deprecated: ``auglib.source.Read``
* Deprecated: ``auglib.Sink``
* Deprecated: ``auglib.sink.Play``
* Deprecated: ``auglib.sink.Write``
* Deprecated: ``auglib.AudioBuffer.play()``
* Removed: ``scipy`` dependency
* Removed: ``humanfriendly`` dependency
* Removed: ``auglib.transform.FilterDesign``
* Removed: ``auglib.transform.ToneShape``
* Removed: ``auglib.ExceptionHandling``
* Removed: ``auglib.LibraryException``
* Removed: ``auglib.LibraryExceptionWarning``


Version 0.8.4 (2021-08-04)
--------------------------

* Changed. Updated underlying binary with latest auglib updates. Main change:
  ClipByRatio transform does not raise exceptions anymore when the computed
  threshold is too low.


Version 0.8.3 (2021-07-20)
--------------------------

* Added: set cache root with ``$AUGLIB_CACHE_ROOT``
* Changed: switched from ``audata`` to ``audformat``


Version 0.8.2 (2020-12-04)
--------------------------

* Added: ``channels`` and ``mixdown`` argument to :class:`auglib.Augment`
* Changed: :class:`auglib.Augment` derives from :class:`audobject.Object`
* Fixed: :meth:`auglib.default_cache_root` uses :meth:`auglib.Augment.id`
* Fixed: restore progress bar in all ``auglib.Augment.process_*`` methods


Version 0.8.1 (2020-11-17)
--------------------------

* Changed: avoid nested progress bar in :meth:`auglib.Augment.augment`


Version 0.8.0 (2020-10-29)
--------------------------

* Added: :class:`auglib.config`, :meth:`auglib.default_cache_root`, :meth:`auglib.clear_default_cache_root`
* Changed: ``cache_root`` argument of :meth:`auglib.Augment.augment` defaults to :meth:`auglib.default_cache_root`


Version 0.7.1 (2020-10-23)
--------------------------

* Changed: remove outdated example from README
* Fixed: remove unused ``as_db`` argument form :class:`auglib.FloatUni`


Version 0.7.0 (2020-10-09)
--------------------------

* Removed: previously deprecated :class:`auglib.AudioModifier` is now removed.


Version 0.6.3 (2020-10-09)
--------------------------

* Changed: Now using the ``audobject`` package to serialize
  :class:`auglib.Transform` objects to YAML.
* Fixed: Safer guards against the usage of negative time values whenever
  ``auglib.core.utils.to_samples`` is called.
* Fixed: Potential bug when using :class:`auglib.Transform.Trim` with a
  ``duration`` member greater than the actual input buffer size.


Version 0.6.2 (2020-10-08)
--------------------------

* Fixed: missing documentation for :class:`auglib.Transform.Trim`


Version 0.6.1 (2020-10-08)
--------------------------

* Added: :class:`auglib.Transform.Trim`.
* Changed: :meth:`auglib.Transform.call` made private (changed into
  :meth:`auglib.Transform._call`).


Version 0.6.0 (2020-09-29)
--------------------------

* Added: :class:`auglib.Augment`
* Changed: use ``audeer`` helper functions
* Changed: implement usage example with ``jupyter-sphinx``
* Changed: mark :class:`auglib.AudioModifier` as deprecated
* Removed: tests for :class:`auglib.AudioModifier`


Version 0.5.3 (2020-09-29)
--------------------------

* Added: documentation on supported bit rates to :class:`auglib.transform.AMRNB`
* Added: link to documentation to Python package


Version 0.5.2 (2020-08-31)
--------------------------

* Added: :class:`auglib.transform.AMRNB`


Version 0.5.1 (2020-07-16)
--------------------------

* Changed: Avoid automatically enabling the ``force_overwrite`` option in
  ``AudioModifier.apply_on_index``. The user is now required to set this
  manually.


Version 0.5.0 (2020-04-24)
--------------------------

* Added: ``compressDynamicRange``: option to restore original peak.


Version 0.4.3 (2020-04-14)
--------------------------

* Fixed: Replace ``utils.mk_dirs`` with ``audeer.mkdir`` (improve thread-safety)


Version 0.4.1 (2020-04-09)
--------------------------

* Added: ``IntList`` and ``FloatList`` as companions to ``StrList``.
* Fixed: ``FloatNorm`` now provided with class members.


Version 0.4.0 (2020-03-04)
--------------------------

* Added: transform ``BandStop``


Version 0.3.8 (2020-02-27)
--------------------------

* Fixed: ``FloatNorm`` properly draws from truncated distribution


Version 0.3.7 (2020-02-26)
--------------------------

* Added: transform ``CompressDynamicRange``
* Changed: remove support for Python 3.5
* Changed: publish package in ci-job


Version 0.3.5 (2020-02-04)
--------------------------

* Changed: allow random filter order


Version 0.3.4 (2020-01-16)
--------------------------

* Changed: allow random filter order


Version 0.3.3 (2020-01-15)
--------------------------

* Added: ``Bool`` and ``BoolRand`` class
* Changed: copyright years


Version 0.3.2 (2019-12-09)
--------------------------

* Fixed: update release instructions to avoid obsolete files in wheel package


Version 0.3.1 (2019-12-09)
--------------------------

* Changed: api documentation with toc-tree


Version 0.3.0 (2019-12-09)
--------------------------

* Added: ``AudioModifier`` interface
* Added: ``relative`` position argument
* Changed: re-structured package


Version 0.2.3 (2019-11-22)
--------------------------

* Added: unit ``relative`` to randomize position relative to buffer length
* Fixed: ``read_pos_dur`` bug in mix function


Version 0.2.2 (2019-11-14)
--------------------------

* Added: handling of exceptions thrown by c library
* Fixed: ``read_pos_aux`` bug in mix function


Version 0.2.1 (2019-11-12)
--------------------------

* Fixed: dependency to ``libSoundTouch.so.1`` is properly resolved if
  called outside the root directory


Version 0.2.0 (2019-11-04)
--------------------------

* Changed: implemented transforms as classes


Version 0.1.5 (2019-10-11)
--------------------------

* Added: ``libSoundTouch`` shared library
* Fixed: high pass filter


Version 0.1.4 (2019-09-30)
--------------------------

* Changed: rely on typehints in docstring


Version 0.1.3 (2019-09-26)
--------------------------

* Added: ``clip_by_ratio()``
* Changed: ``Tone`` constructor
* Changed: ``[low,high,band]_pass()`` arguments
* Changed: ``clip()`` arguments


Version 0.1.2 (2019-09-23)
--------------------------

* Added: add icon


Version 0.1.1 (2019-09-23)
--------------------------

* Added: advanced usage example
* Added: ``AudioBuffer.from_file()`` to read from an audio file
* Added: ``AudioBuffer.to_file()`` to save buffer to a an audio file
* Changed: ``AudioBuffer.FromArray()`` to ``AudioBuffer.from_array()``


Version 0.1.0 (2019-09-08)
--------------------------

* Added: initial release


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
