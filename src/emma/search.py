"""Vectorised search and filtering engine for EMMa.

Provides :class:`SearchQuery` (an immutable filter specification) and
:class:`EmmaSearcher` (the execution engine).  All heavy lifting is done
with NumPy boolean masks; no per-item Python loops run during filtering.

Usage
-----
>>> from emma import EmmaDataset, EmmaSearcher, SearchQuery
>>>
>>> ds = EmmaDataset("/data/emma")
>>> s  = EmmaSearcher(ds)                        # or EmmaSearcher.from_root(...)
>>>
>>> # Keyword shorthand (most common)
>>> asins = s.query_asins(splits=["train"], price_range=(5, 50), limit=100)
>>>
>>> # Reusable query object
>>> q = SearchQuery(splits=("train",), materials=("metal", "plastic"), materials_and=True)
>>> n = s.count(q)
>>> records = s.query_records(q, limit=20)       # override just *limit*
>>>
>>> # Progress callback
>>> s.query_asins(q, progress=lambda stage, n: print(f"{stage}: {n:,}"))
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _constants as C
from .dataset import EmmaDataset

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

ProgressFn = Callable[[str, int], None]
"""Signature: ``(stage_name, remaining_candidate_count) -> None``."""

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_EMPTY = np.empty(0, dtype=np.int32)

# User-friendly aliases → attribute names on ``ListedFlags``.
_LISTED_ALIASES: dict[str, str] = {
    "category": "category_listed",
    "cat": "category_listed",
    "price": "price_listed",
    "cost": "price_listed",
    "mass": "mass_listed",
    "weight": "mass_listed",
}

# User-friendly aliases → attribute names on ``AvailabilityFlags``.
_PRESENT_ALIASES: dict[str, str] = {
    "package_dimensions": "package_dimensions_present",
    "package_dimension": "package_dimensions_present",
    "dimensions": "package_dimensions_present",
    "dimension": "package_dimensions_present",
    "dims": "package_dimensions_present",
}

_VALID_LISTED_ATTRS = sorted(set(_LISTED_ALIASES.values()))
_VALID_PRESENT_ATTRS = sorted(set(_PRESENT_ALIASES.values()))
_VALID_TEXT_FIELDS = ("title", "description", "features")

_TEXT_FIELD_KEYS: dict[str, str] = {
    "title": C.RecordField.TITLE,
    "description": C.RecordField.DESCRIPTION,
    "features": C.RecordField.FEATURES,
}


# ---------------------------------------------------------------------------
# SearchQuery
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Immutable, reusable description of a search over the EMMa dataset.

    Every filter field defaults to *no constraint*.  Create variants with
    :func:`dataclasses.replace`::

        base = SearchQuery(splits=("train",))
        cheap = dataclasses.replace(base, price_range=(0, 10))

    Parameters
    ----------
    splits : sequence of str, optional
        Keep items belonging to **any** of these splits (OR).
    price_range : (lo, hi), optional
        Inclusive price bounds; ``None`` on either side for one-sided.
    mass_range : (lo, hi), optional
        Inclusive mass/weight bounds; same convention as *price_range*.
    quantity_range : (lo, hi), optional
        Inclusive quantity bounds; ``None`` on either side for one-sided.
    require_listed : sequence of str
        Fields that must be *listed* (not estimated).
        Recognised aliases: ``"category"``/``"cat"``,
        ``"price"``/``"cost"``, ``"mass"``/``"weight"``.
    require_present : sequence of str
        Fields that must be present in the source data but are not part of
        the listed-vs-EM bitfield.  Recognised aliases:
        ``"package_dimensions"``/``"dimensions"``/``"dims"``.
    require_custom : sequence of str
        Custom boolean attributes that must be ``True``
        (e.g. ``"sharp"``, ``"transparent"``).
    materials : sequence of str
        Material names to filter on.
    materials_and : bool
        ``True`` → item must match **all** materials;
        ``False`` (default) → **any** suffices.
    categories : sequence of (id, depth)
        Explicit category keys and all their descendants (OR).
    category_names : sequence of str
        Case-insensitive substring search against known category names.
        Matched keys are unioned with *categories*.
    text_query : str, optional
        Literal string or regex pattern applied to selected text fields.
    text_fields : sequence of str
        Any of ``"title"``, ``"description"``, ``"features"``.
        Empty means all three fields.
    text_regex : bool
        ``True`` → interpret *text_query* as a regular expression.
    limit : int, optional
        Maximum results returned.
    sort_by : ``"sort_priority"`` or ``None``
        ``None`` skips sorting entirely (faster for counts).
    descending : bool
        Sort direction when *sort_by* is set (highest priority first
        when ``True``).
    """

    splits: Sequence[str] | None = None
    price_range: tuple[float | None, float | None] | None = None
    mass_range: tuple[float | None, float | None] | None = None
    quantity_range: tuple[int | None, int | None] | None = None
    require_listed: Sequence[str] = ()
    require_present: Sequence[str] = ()
    require_custom: Sequence[str] = ()
    materials: Sequence[str] = ()
    materials_and: bool = False
    categories: Sequence[tuple[int, int]] = ()
    category_names: Sequence[str] = ()
    text_query: str | None = None
    text_fields: Sequence[str] = ()
    text_regex: bool = False
    limit: int | None = None
    sort_by: str | None = "sort_priority"
    descending: bool = True


# ---------------------------------------------------------------------------
# EmmaSearcher
# ---------------------------------------------------------------------------


class EmmaSearcher:
    """Vectorised filter-and-sort engine over an :class:`EmmaDataset`.

    Uses **composition**: pass an already-loaded dataset, or use the
    convenience constructor :meth:`from_root` to build one automatically.

    Every public ``query_*`` method accepts **either** a :class:`SearchQuery`
    instance **or** keyword arguments (forwarded to ``SearchQuery(...)``).
    You may also combine both — keywords override fields of the query::

        base = SearchQuery(splits=("train",))
        searcher.query_asins(base, limit=50)   # adds limit on top
    """

    __slots__ = ("_ds",)

    def __init__(self, dataset: EmmaDataset) -> None:
        self._ds = dataset

    # ── Factories ────────────────────────────────────────────────────

    @classmethod
    def from_root(cls, root: str | Path = ".") -> EmmaSearcher:
        """Build the underlying :class:`EmmaDataset`, then wrap it."""
        return cls(EmmaDataset(root))

    # ── Properties ───────────────────────────────────────────────────

    @property
    def dataset(self) -> EmmaDataset:
        """The wrapped :class:`EmmaDataset` instance."""
        return self._ds

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dataset={self._ds!r})"

    # ── Public query entry-points ────────────────────────────────────

    def query_indices(
        self,
        query: SearchQuery | None = None,
        *,
        progress: ProgressFn | None = None,
        **overrides: Any,
    ) -> np.ndarray:
        """Return matching row indices as an ``int32`` NumPy array.

        Parameters
        ----------
        query : SearchQuery, optional
            Pre-built query.  If omitted, one is constructed from
            *overrides*.
        progress : callable, optional
            ``(stage_name, remaining_count)`` callback invoked after
            each filter stage.
        **overrides
            Keyword arguments forwarded to :class:`SearchQuery`, or used
            to patch *query* via :func:`dataclasses.replace`.
        """
        return self._execute(self._resolve(query, overrides), progress)

    def query_asins(
        self,
        query: SearchQuery | None = None,
        *,
        progress: ProgressFn | None = None,
        **overrides: Any,
    ) -> list[str]:
        """Return matching ASINs (no FDD record lookups).

        Accepts the same parameters as :meth:`query_indices`.
        """
        idxs = self._execute(self._resolve(query, overrides), progress)
        return self._ds.asins[idxs].tolist()

    def query_records(
        self,
        query: SearchQuery | None = None,
        *,
        progress: ProgressFn | None = None,
        **overrides: Any,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(asin, record_dict)`` pairs for every match.

        Accepts the same parameters as :meth:`query_indices`.
        """
        idxs = self._execute(self._resolve(query, overrides), progress)
        ds = self._ds
        return [(a, ds[a]) for a in ds.asins[idxs]]

    def count(
        self,
        query: SearchQuery | None = None,
        *,
        progress: ProgressFn | None = None,
        **overrides: Any,
    ) -> int:
        """Count matching items *without* sorting or materialising results.

        Automatically forces ``sort_by=None`` and ``limit=None`` for speed.
        """
        q = self._resolve(query, overrides)
        q = dataclasses.replace(q, sort_by=None, limit=None)
        return int(self._execute(q, progress).size)

    # ── Internal: query resolution ───────────────────────────────────

    @staticmethod
    def _resolve(
        query: SearchQuery | None,
        overrides: dict[str, Any],
    ) -> SearchQuery:
        """Merge *query* and *overrides* into a single :class:`SearchQuery`."""
        if query is not None and overrides:
            return dataclasses.replace(query, **overrides)
        if query is not None:
            return query
        return SearchQuery(**overrides)

    # ── Internal: filter pipeline ────────────────────────────────────

    def _execute(
        self,
        q: SearchQuery,
        progress: ProgressFn | None,
    ) -> np.ndarray:
        """Run the full filter → sort → limit pipeline."""
        ds = self._ds
        n = len(ds)
        mask = np.ones(n, dtype=bool)

        def _alive(stage: str) -> bool:
            """Report progress; return *False* when no candidates remain."""
            if progress is not None:
                progress(stage, int(mask.sum()))
            return mask.any()

        # 1 ── Splits (OR) ───────────────────────────────────────────
        if q.splits:
            acc = np.zeros(n, dtype=bool)
            for name in q.splits:
                split = ds.split_masks.get(name)
                if split is None:
                    raise ValueError(f"Unknown split: {name!r}")
                acc |= split
            mask &= acc
        if not _alive("splits"):
            return _EMPTY

        # 2 ── Categories (OR, with optional substring matching) ─────
        cat_keys = self._collect_category_keys(q)
        if cat_keys:
            acc = np.zeros(n, dtype=bool)
            for key in cat_keys:
                idxs = ds.category_indices.get(key)
                if idxs is not None and idxs.size:
                    acc[idxs] = True
            mask &= acc
        if not _alive("categories"):
            return _EMPTY

        # 3 ── Listed requirements (AND) ─────────────────────────────
        for field_name in q.require_listed:
            attr = _LISTED_ALIASES.get(field_name.lower())
            if attr is None:
                raise ValueError(
                    f"Unknown listed field: {field_name!r}. "
                    f"Valid attributes: {_VALID_LISTED_ATTRS}"
                )
            mask &= getattr(ds.listed, attr)
        if not _alive("listed"):
            return _EMPTY

        # 4 ── Presence requirements (AND) ───────────────────────────
        for field_name in q.require_present:
            attr = _PRESENT_ALIASES.get(field_name.lower())
            if attr is None:
                raise ValueError(
                    f"Unknown present field: {field_name!r}. "
                    f"Valid attributes: {_VALID_PRESENT_ATTRS}"
                )
            field_mask = getattr(ds.available, attr)
            if field_mask is None:
                raise ValueError(
                    f"Presence mask for {field_name!r} is not available. "
                    "Generate the corresponding search file first.",
                )
            mask &= field_mask
        if not _alive("present"):
            return _EMPTY

        # 5 ── Custom boolean attributes (AND) ───────────────────────
        for name in q.require_custom:
            try:
                idx = ds.custom_attribute_index(name)
            except KeyError:
                raise ValueError(f"Unknown custom attribute: {name!r}") from None
            mask &= ds.custom_bits[:, idx]
        if not _alive("custom"):
            return _EMPTY

        # 6 ── Materials (AND / OR) ──────────────────────────────────
        if q.materials:
            try:
                midxs = np.array(
                    [ds.material_index(m) for m in q.materials],
                    dtype=np.intp,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from None
            mat_cols = ds.material_bits[:, midxs]  # (N, k)
            combine = np.ndarray.all if q.materials_and else np.ndarray.any
            mask &= combine(mat_cols, axis=1)
        if not _alive("materials"):
            return _EMPTY

        # 7 ── Price range ───────────────────────────────────────────
        if q.price_range is not None:
            lo, hi = q.price_range
            if lo is not None:
                mask &= ds.price >= lo
            if hi is not None:
                mask &= ds.price <= hi
        if not _alive("price"):
            return _EMPTY

        # 8 ── Mass range ────────────────────────────────────────────
        if q.mass_range is not None:
            lo, hi = q.mass_range
            if lo is not None:
                mask &= ds.mass >= lo
            if hi is not None:
                mask &= ds.mass <= hi
        if not _alive("mass"):
            return _EMPTY

        # 9 ── Quantity range ───────────────────────────────────────
        if q.quantity_range is not None:
            lo, hi = q.quantity_range
            if lo is not None:
                mask &= ds.quantity >= lo
            if hi is not None:
                mask &= ds.quantity <= hi
        if not _alive("quantity"):
            return _EMPTY

        # 10 ── Text search (literal / regex) ───────────────────────
        if q.text_query is not None:
            fields = q.text_fields or _VALID_TEXT_FIELDS
            invalid = sorted(set(fields) - set(_VALID_TEXT_FIELDS))
            if invalid:
                raise ValueError(
                    f"Unknown text field(s): {invalid!r}. "
                    f"Valid fields: {_VALID_TEXT_FIELDS}"
                )
            candidates = np.flatnonzero(mask)
            if q.text_regex:
                try:
                    match = re.compile(q.text_query, re.IGNORECASE).search
                except re.error as exc:
                    raise ValueError(
                        f"Invalid text regex: {q.text_query!r}"
                    ) from exc
            else:
                needle = q.text_query.casefold()

                def match(s: str) -> bool:
                    return needle in s.casefold()

            keep = np.fromiter(
                (
                    any(
                        match(
                            "\n".join(rec[_TEXT_FIELD_KEYS[field]])
                            if field == "features"
                            else rec[_TEXT_FIELD_KEYS[field]]
                        )
                        for field in fields
                    )
                    for rec in (ds[a] for a in ds.asins[candidates])
                ),
                dtype=bool,
                count=candidates.size,
            )
            mask[:] = False
            mask[candidates[keep]] = True
        if not _alive("text"):
            return _EMPTY

        # 11 ── Sort & limit ─────────────────────────────────────────
        result = self._sort_and_limit(mask, q.sort_by, q.descending, q.limit)
        _alive("done")
        return result

    # ── Internal: category key collection ────────────────────────────

    def _collect_category_keys(
        self,
        q: SearchQuery,
    ) -> set[tuple[int, int]]:
        """Union explicit *categories* with substring-matched *category_names*."""
        keys: set[tuple[int, int]] = set(q.categories)
        if q.category_names:
            lowered = [s.lower() for s in q.category_names]
            for (cid, depth), cname in self._ds.cat_id_to_name.items():
                cname_lower = cname.lower()
                if any(tok in cname_lower for tok in lowered):
                    keys.add((cid, depth))
        return {
            descendant
            for key in keys
            for descendant in self._ds.category_subtrees.get(key, (key,))
        }

    # ── Internal: sort & limit ───────────────────────────────────────

    def _sort_and_limit(
        self,
        mask: np.ndarray,
        sort_by: str | None,
        descending: bool,
        limit: int | None,
    ) -> np.ndarray:
        ds = self._ds

        # No sorting requested
        if sort_by is None:
            idxs = np.flatnonzero(mask)
            if limit is not None:
                idxs = idxs[:limit]
            return idxs.astype(np.int32, copy=False)

        if sort_by != "sort_priority":
            raise ValueError(
                f"Unsupported sort_by={sort_by!r}; "
                f"use 'sort_priority' or None."
            )

        # Fast path: every row matches → use the pre-sorted array directly
        if mask.all():
            ordered = ds.sorted_indices if descending else ds.sorted_indices[::-1]
            if limit is not None:
                ordered = ordered[:limit]
            return ordered.astype(np.int32, copy=False)

        # General path: sort only the surviving candidates by rank
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return _EMPTY

        ranks = ds.priority_rank[candidates]
        order = np.argsort(ranks)  # ascending rank = descending score
        if not descending:
            order = order[::-1]
        if limit is not None:
            order = order[:limit]
        return candidates[order].astype(np.int32, copy=False)
