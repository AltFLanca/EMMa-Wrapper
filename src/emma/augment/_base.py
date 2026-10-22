"""Shared augmentation interfaces and lightweight in-memory sources."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from typing import Any


class AugmentSource(abc.ABC):
    """Read-only source of per-item augmentation data.

    Subclass this to plug in a custom storage backend. Concrete
    implementations must provide :pyattr:`name`, :meth:`__getitem__`,
    :meth:`__contains__`, :meth:`__len__`, and :meth:`keys`.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this augmentation (e.g. ``"point_cloud"``)."""

    @abc.abstractmethod
    def __getitem__(self, asin: str) -> Any:
        """Retrieve data for *asin*. Raise ``KeyError`` if absent."""

    @abc.abstractmethod
    def __contains__(self, asin: str) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        """ASINs for which data is available."""

    def get(self, asin: str, default: Any = None) -> Any:
        """Return data for *asin*, or *default* if absent."""
        if asin in self:
            return self[asin]
        return default

    def get_batch(self, asins: Sequence[str]) -> dict[str, Any]:
        """Retrieve multiple items. Override for batch-optimised I/O."""
        return {a: self[a] for a in asins if a in self}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, n_items={len(self):,})"


class InMemorySource(AugmentSource):
    """``dict``-backed source for computed or transient data.

    Supports ``__setitem__`` and ``__delitem__`` for mutation.

    Parameters
    ----------
    name : str
        Augmentation name.
    data : dict, optional
        Initial contents (shallow-copied).
    """

    __slots__ = ("_name", "_data")

    def __init__(self, name: str, data: dict[str, Any] | None = None) -> None:
        self._name = name
        self._data: dict[str, Any] = dict(data) if data else {}

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, asin: str) -> Any:
        return self._data[asin]

    def __contains__(self, asin: str) -> bool:
        return asin in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def __setitem__(self, asin: str, value: Any) -> None:
        self._data[asin] = value

    def __delitem__(self, asin: str) -> None:
        del self._data[asin]
