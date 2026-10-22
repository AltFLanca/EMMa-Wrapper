"""EMMa dataset wrapper package."""

from .augment import (
    AugmentedDataset,
    AugmentSource,
    DirectorySource,
    FDDSource,
    HDF5ArraySource,
    HDF5FieldSpec,
    HDF5Source,
    InMemorySource,
)
from .dataset import EmmaDataset
from .search import EmmaSearcher, SearchQuery
from .types import EMFlags, Ratings

__all__ = [
    "AugmentedDataset",
    "AugmentSource",
    "DirectorySource",
    "EMFlags",
    "EmmaDataset",
    "EmmaSearcher",
    "FDDSource",
    "HDF5ArraySource",
    "HDF5FieldSpec",
    "HDF5Source",
    "InMemorySource",
    "Ratings",
    "SearchQuery",
]
