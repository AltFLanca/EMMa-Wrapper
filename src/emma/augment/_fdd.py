"""Freeze-Dried-Data-backed augmentation sources."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from freeze_dried_data.freeze_dried_data_old import FDD

from ._base import AugmentSource


class FDDSource(AugmentSource):
    """Wrap an existing Freeze-Dried-Data archive.

    Parameters
    ----------
    name : str
        Augmentation name.
    path : str or Path
        Path to the ``.fdd`` file.
    """

    __slots__ = ("_name", "_fdd", "_keys")

    def __init__(self, name: str, path: str | Path) -> None:
        self._name = name
        self._fdd = FDD(str(path))
        self._keys: frozenset[str] = frozenset(self._fdd.keys())

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, asin: str) -> Any:
        if asin not in self._keys:
            raise KeyError(asin)
        return self._fdd[asin]

    def __contains__(self, asin: str) -> bool:
        return asin in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self) -> Iterable[str]:
        return self._keys
