"""Bitfield decoders for EMMa search arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._constants import ListedBit, NUM_CUSTOM_BITS


@dataclass(frozen=True, slots=True)
class ListedFlags:
    """Boolean masks: ``True`` → value was *listed* (not estimated).

    Attributes
    ----------
    category_listed, price_listed, mass_listed : np.ndarray[bool]
        Per-item flags derived from the listed-search byte.
    """

    category_listed: np.ndarray
    price_listed: np.ndarray
    mass_listed: np.ndarray

    # Convenience inverses (cheap vectorised negation, not cached).
    @property
    def category_em(self) -> np.ndarray:
        return ~self.category_listed

    @property
    def price_em(self) -> np.ndarray:
        return ~self.price_listed

    @property
    def mass_em(self) -> np.ndarray:
        return ~self.mass_listed


@dataclass(frozen=True, slots=True)
class AvailabilityFlags:
    """Boolean masks describing whether non-EM fields are present.

    Attributes
    ----------
    package_dimensions_present : np.ndarray[bool] or None
        ``True`` where package dimensions are available in the record.
        This is intentionally distinct from :class:`ListedFlags`: package
        dimensions are not part of the listed-vs-EM bitfield.
    """

    package_dimensions_present: np.ndarray | None = None

    @property
    def package_dimensions_missing(self) -> np.ndarray:
        if self.package_dimensions_present is None:
            raise AttributeError(
                "package_dimensions_present mask has not been loaded.",
            )
        return ~self.package_dimensions_present


def decode_listed(raw: np.ndarray) -> ListedFlags:
    """Unpack a uint8 column into three boolean flag arrays.

    Bit layout (MSB-first after ``np.unpackbits``):
    ``0 → category listed  |  1 → price listed  |  2 → mass listed``
    """
    bits = np.unpackbits(raw.astype(np.uint8).reshape(-1, 1), axis=1)
    return ListedFlags(
        category_listed=bits[:, ListedBit.CATEGORY].astype(bool),
        price_listed=bits[:, ListedBit.PRICE].astype(bool),
        mass_listed=bits[:, ListedBit.MASS].astype(bool),
    )



def decode_availability_mask(
    raw: np.ndarray | None,
    *,
    n_items: int,
) -> AvailabilityFlags:
    """Normalise a stored presence mask into :class:`AvailabilityFlags`.

    Accepts either a boolean array of length ``n_items`` or a packed uint8
    bit-vector.  ``None`` means the optional mask file is absent.
    """
    if raw is None:
        return AvailabilityFlags(package_dimensions_present=None)

    arr = np.asarray(raw)
    if arr.dtype == np.bool_:
        mask = arr.ravel()
    else:
        arr = arr.astype(np.uint8).ravel()
        mask = arr.astype(bool) if arr.size == n_items else np.unpackbits(arr)
        mask = mask[:n_items].astype(bool, copy=False)

    if mask.size != n_items:
        raise ValueError(
            "Invalid package-dimensions presence mask length: "
            f"expected {n_items}, got {mask.size}.",
        )

    return AvailabilityFlags(package_dimensions_present=mask)



def decode_custom(raw: np.ndarray) -> np.ndarray:
    """Unpack custom-attribute byte → ``(N, NUM_CUSTOM_BITS)`` bool array.

    Bit order: sharp, transparent, deformable, articulating, electronic.
    """
    bits = np.unpackbits(raw.astype(np.uint8).reshape(-1, 1), axis=1)
    return bits[:, :NUM_CUSTOM_BITS].astype(bool)



def decode_materials(raw: np.ndarray, num_materials: int) -> np.ndarray:
    """Unpack material byte-array → ``(N, num_materials)`` bool matrix."""
    bits = np.unpackbits(raw.astype(np.uint8), axis=1)
    return bits[:, :num_materials].astype(bool)
