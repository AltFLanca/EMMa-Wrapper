"""Augmentation framework for EMMa.

Provides a plugin system for attaching additional per-item data
(point clouds, depth maps, segmentation masks, ...) to the base dataset
without modifying it.

Storage backends
----------------
:class:`DirectorySource`
    One file per item (``{directory}/{asin}.npy`` etc.).  Best for
    moderate-count, large-per-item data such as point clouds.

:class:`HDF5ArraySource`
    Single HDF5 file with one row per ASIN and one or more dense fields.
    Best for resumable, compressed, uniform-shape outputs such as depth
    maps and embeddings.  Requires ``h5py``.

:class:`HDF5Source`
    Single HDF5 file keyed by ASIN.  Kept for simpler one-dataset-per-item
    workflows.

:class:`FDDSource`
    Wraps an existing Freeze-Dried-Data archive.

:class:`InMemorySource`
    Plain ``dict`` -- useful for computed or transient augmentations.

Quick start
-----------
>>> from emma import EmmaDataset, AugmentedDataset
>>> from emma.augment import DirectorySource
>>>
>>> ds = EmmaDataset("/data/emma")
>>> ads = AugmentedDataset(ds)
>>>
>>> pcs = DirectorySource("point_cloud", "/data/emma/pointclouds")
>>> ads.register(pcs)
>>>
>>> rec = ads.get_record(asin)
>>> pc = ads.point_cloud[asin]            # attribute-style access
>>> pc = ads.get_augmentation("point_cloud", asin)   # explicit access
>>> full = ads.get_augmented_record(asin)   # base record + augmentations
"""

from ._base import AugmentSource, InMemorySource
from ._dataset import AugmentedDataset
from ._directory import DirectorySource
from ._fdd import FDDSource
from ._hdf5 import HDF5ArraySource, HDF5FieldSpec, HDF5Source

__all__ = [
    "AugmentedDataset",
    "AugmentSource",
    "DirectorySource",
    "FDDSource",
    "HDF5ArraySource",
    "HDF5FieldSpec",
    "HDF5Source",
    "InMemorySource",
]
