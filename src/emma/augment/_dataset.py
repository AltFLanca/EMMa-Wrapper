"""Augmented dataset wrapper and source registration logic."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..dataset import EmmaDataset
from ._base import AugmentSource


class AugmentedDataset:
    """Compose an :class:`EmmaDataset` with augmentation sources.

    Every :class:`EmmaDataset` attribute and method is accessible directly
    through delegation. Registered sources can be reached by name via
    attribute access (``ads.point_cloud[asin]``), or explicitly via
    :meth:`get_augmentation`.

    Parameters
    ----------
    dataset : EmmaDataset

    Notes
    -----
    If a source name shadows an ``EmmaDataset`` attribute, the source
    takes priority when accessed as an attribute. Use
    ``ads.dataset.<attr>`` to reach the shadowed attribute, or
    ``ads.source(<name>)`` to unambiguously access the source.
    """

    __slots__ = ("_ds", "_sources")

    def __init__(self, dataset: EmmaDataset) -> None:
        self._ds = dataset
        self._sources: dict[str, AugmentSource] = {}

    @classmethod
    def from_root(
        cls,
        root: str | Path = ".",
        sources: Iterable[AugmentSource] = (),
    ) -> AugmentedDataset:
        """Build dataset and register sources in one call."""
        ads = cls(EmmaDataset(root))
        for src in sources:
            ads.register(src)
        return ads

    def register(self, source: AugmentSource) -> AugmentedDataset:
        """Add a source. Returns ``self`` for chaining."""
        name = source.name
        if name in self._sources:
            raise ValueError(f"Source {name!r} is already registered.")
        if hasattr(self._ds, name):
            warnings.warn(
                f"Source name {name!r} shadows an EmmaDataset attribute. "
                f"Use .dataset.{name} or .source({name!r}) to disambiguate.",
                stacklevel=2,
            )
        self._sources[name] = source
        return self

    def unregister(self, name: str) -> AugmentSource:
        """Remove and return a source by name."""
        try:
            return self._sources.pop(name)
        except KeyError:
            raise KeyError(f"No source registered as {name!r}.") from None

    def source(self, name: str) -> AugmentSource:
        """Retrieve a source by name (explicit, unambiguous)."""
        try:
            return self._sources[name]
        except KeyError:
            raise KeyError(
                f"No source registered as {name!r}. "
                f"Available: {self.source_names}",
            ) from None

    @property
    def source_names(self) -> list[str]:
        """Names of all registered sources."""
        return list(self._sources)

    @property
    def dataset(self) -> EmmaDataset:
        """The underlying :class:`EmmaDataset`."""
        return self._ds

    def get_augmentation(self, name: str, asin: str) -> Any:
        """Retrieve data from the named source for *asin*."""
        return self.source(name)[asin]

    def has_augmentation(self, name: str, asin: str) -> bool:
        """Check whether the named source has data for *asin*."""
        src = self._sources.get(name)
        return src is not None and asin in src

    def get_augmented_record(
        self,
        asin: str,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Base record dict with available augmentation fields merged in."""
        rec = dict(self._ds.get_record(asin))
        sources = (
            self._sources
            if include is None
            else {k: self._sources[k] for k in include}
        )
        for src_name, src in sources.items():
            if asin in src:
                rec[src_name] = src[asin]
        return rec

    def __getattr__(self, name: str) -> Any:
        sources = object.__getattribute__(self, "_sources")
        if name in sources:
            return sources[name]
        ds = object.__getattribute__(self, "_ds")
        return getattr(ds, name)

    def __getitem__(self, asin: str) -> dict[str, Any]:
        return self._ds[asin]

    def __len__(self) -> int:
        return len(self._ds)

    def __contains__(self, asin: str) -> bool:
        return asin in self._ds

    def __repr__(self) -> str:
        srcs = ", ".join(f"{s.name}({len(s):,})" for s in self._sources.values())
        return (
            f"{type(self).__name__}("
            f"dataset={self._ds!r}, "
            f"sources=[{srcs}])"
        )

    def close(self) -> None:
        """Close every source that has a ``close`` method (e.g. HDF5)."""
        for src in self._sources.values():
            close_fn = getattr(src, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> AugmentedDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
