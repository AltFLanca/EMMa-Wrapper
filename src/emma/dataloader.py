"""PyTorch :class:`~torch.utils.data.Dataset` adapter for EMMa."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from .dataset import EmmaDataset

# ── Collation ────────────────────────────────────────────────────────────


def _try_collate(values: list[Any]) -> Any:
    """Stack *values* into a tensor when possible, else return as-is."""
    try:
        return default_collate(values)
    except (RuntimeError, TypeError, ValueError, KeyError):
        return values


def emma_collate(batch: list[Any]) -> Any:
    """Collate function tolerant of variable-length fields.

    Behaves identically to :func:`torch.utils.data.default_collate` for
    numeric scalars, tensors, and fixed-shape arrays.  Variable-length
    sequences (e.g. feature bullet-points, category hierarchies) that
    would otherwise raise are collected into plain Python lists instead.

    Designed for dict-valued batches returned by :class:`EmmaMapDataset`,
    but safe for any structure accepted by *default_collate*.

    Examples
    --------
    >>> loader = DataLoader(emma_ds, batch_size=32, collate_fn=emma_collate)
    """
    out = {}

    for key in batch[0]:
        values = [sample[key] for sample in batch]

        if key == "images":
            # keep ragged / per-sample image collections as a list
            out[key] = values
        else:
            try:
                out[key] = default_collate(values)
            except (RuntimeError, TypeError, ValueError, KeyError):
                out[key] = values

    return out


# ── Map-style dataset ───────────────────────────────────────────────────


class EmmaMapDataset(Dataset):
    """Map-style :class:`~torch.utils.data.Dataset` over selected ASINs.

    Wraps an :class:`EmmaDataset` and exposes a subset defined by *asins*
    (typically obtained from :meth:`~search.EmmaSearcher.query_asins`).
    An optional *transform* converts each ``(asin, record)`` pair into
    the format required by your training loop.

    Parameters
    ----------
    dataset : EmmaDataset
        The loaded EMMa dataset.
    asins : sequence of str
        ASINs to expose (order is preserved).
    transform : callable, optional
        ``(asin: str, record: dict) -> Any`` applied per sample.
        When ``None``, ``__getitem__`` returns ``(asin, record)``.

    Examples
    --------
    >>> from torch.utils.data import DataLoader
    >>> from emma import EmmaDataset, EmmaSearcher
    >>> from emma.dataloader import EmmaMapDataset, emma_collate
    >>>
    >>> ds = EmmaDataset("/data/emma")
    >>> searcher = EmmaSearcher(ds)
    >>> train_asins = searcher.query_asins(
    ...     splits=["train"], require_listed=["mass"],
    ... )
    >>> torch_ds = EmmaMapDataset(ds, train_asins, transform=my_fn)
    >>> loader = torch_ds.to_dataloader(batch_size=32, shuffle=True)
    """

    __slots__ = ("_ds", "_asins", "_transform")

    def __init__(
        self,
        dataset: EmmaDataset,
        asins: Sequence[str],
        *,
        transform: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._ds = dataset
        self._asins = list(asins)
        self._transform = transform

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._asins)

    def __getitem__(self, index: int) -> Any:
        asin = self._asins[index]
        rec = self._ds[asin]
        if self._transform is not None:
            return self._transform(asin, rec)
        return asin, rec

    # ── Convenience ──────────────────────────────────────────────────

    def to_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        *,
        collate_fn: Callable[[list[Any]], Any] = emma_collate,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a :class:`~torch.utils.data.DataLoader` for this subset.

        Defaults to :func:`emma_collate`, which gracefully handles
        variable-length fields such as feature lists and category
        hierarchies.  All extra keyword arguments are forwarded to
        :class:`~torch.utils.data.DataLoader`.

        Parameters
        ----------
        batch_size : int
        shuffle : bool
        collate_fn : callable, optional
            Defaults to :func:`emma_collate`.
        **kwargs
            Forwarded to :class:`~torch.utils.data.DataLoader`.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    # ── Properties ───────────────────────────────────────────────────

    @property
    def dataset(self) -> EmmaDataset:
        """The underlying :class:`EmmaDataset`."""
        return self._ds

    @property
    def asins(self) -> list[str]:
        """The ASIN list defining this subset (read-only copy)."""
        return list(self._asins)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_items={len(self):,}, "
            f"transform={'set' if self._transform else 'None'})"
        )