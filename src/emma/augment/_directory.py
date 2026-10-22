"""Directory-backed augmentation sources."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ._base import AugmentSource


class DirectorySource(AugmentSource):
    """One file per item: ``{directory}/{asin}{suffix}``.

    The directory is scanned at construction time to build an in-memory
    index. Set *scan* to ``False`` for lazy ``stat``-based lookups when
    the number of files is very large.

    Parameters
    ----------
    name : str
        Augmentation name.
    directory : str or Path
        Root directory.
    suffix : str
        File extension (determines the default loader / saver).
    loader : callable, optional
        ``(Path) -> data``. Inferred from *suffix* when omitted.
    saver : callable, optional
        ``(Path, data) -> None``. Required for :meth:`save` with
        non-standard suffixes.
    scan : bool
        ``True`` (default) pre-builds an in-memory ASIN index;
        ``False`` checks the filesystem on each access.

    Supported default suffixes: ``.npy``, ``.npz``, ``.pkl``.
    """

    _DEFAULT_LOADERS: dict[str, Callable[[Path], Any]] = {
        ".npy": lambda p: np.load(p),
        ".npz": lambda p: np.load(p),
        ".pkl": lambda p: pickle.loads(p.read_bytes()),
    }

    _DEFAULT_SAVERS: dict[str, Callable[[Path, Any], None]] = {
        ".npy": lambda p, d: np.save(p, d),
        ".npz": lambda p, d: (
            np.savez(p, **d) if isinstance(d, dict) else np.savez(p, data=d)
        ),
        ".pkl": lambda p, d: p.write_bytes(pickle.dumps(d)),
    }

    __slots__ = ("_name", "_dir", "_suffix", "_loader", "_saver", "_index")

    def __init__(
        self,
        name: str,
        directory: str | Path,
        suffix: str = ".npy",
        loader: Callable[[Path], Any] | None = None,
        saver: Callable[[Path, Any], None] | None = None,
        *,
        scan: bool = True,
    ) -> None:
        self._name = name
        self._dir = Path(directory)
        self._suffix = suffix
        self._loader = loader or self._DEFAULT_LOADERS.get(suffix)
        self._saver = saver or self._DEFAULT_SAVERS.get(suffix)

        if self._loader is None:
            raise ValueError(
                f"No default loader for {suffix!r}. "
                f"Provide a custom 'loader' callable.",
            )

        self._index: dict[str, Path] | None
        if scan and self._dir.exists():
            self._index = {p.stem: p for p in self._dir.glob(f"*{suffix}")}
        elif scan:
            self._index = {}
        else:
            self._index = None

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, asin: str) -> Any:
        path = self._resolve(asin)
        if path is None:
            raise KeyError(asin)
        return self._loader(path)

    def __contains__(self, asin: str) -> bool:
        return self._resolve(asin) is not None

    def __len__(self) -> int:
        if self._index is not None:
            return len(self._index)
        if not self._dir.exists():
            return 0
        return sum(1 for _ in self._dir.glob(f"*{self._suffix}"))

    def keys(self) -> Iterable[str]:
        if self._index is not None:
            return self._index.keys()
        if not self._dir.exists():
            return iter(())
        return (p.stem for p in self._dir.glob(f"*{self._suffix}"))

    def save(self, asin: str, data: Any) -> None:
        """Persist *data* for *asin* and update the index."""
        if self._saver is None:
            raise RuntimeError(
                f"No saver for {self._suffix!r}. "
                f"Provide a custom 'saver' callable.",
            )
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{asin}{self._suffix}"
        self._saver(path, data)
        if self._index is not None:
            self._index[asin] = path

    def refresh_index(self) -> None:
        """Re-scan the directory (only meaningful when *scan=True*)."""
        if self._index is None:
            return
        if self._dir.exists():
            self._index = {p.stem: p for p in self._dir.glob(f"*{self._suffix}")}
        else:
            self._index.clear()

    def _resolve(self, asin: str) -> Path | None:
        if self._index is not None:
            return self._index.get(asin)
        path = self._dir / f"{asin}{self._suffix}"
        return path if path.is_file() else None
