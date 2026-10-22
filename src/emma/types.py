"""Public data types returned by EmmaDataset accessor methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Ratings:
    """Star-rating distribution and total review count.

    Attributes
    ----------
    distribution : tuple[float, ...]
        Fraction of reviews at each star level, ordered 1★ → 5★.
    count : int
        Total number of reviews.
    """

    distribution: tuple[float, ...]
    count: int


@dataclass(frozen=True, slots=True)
class EMFlags:
    """Per-record flags indicating which fields were estimated via EM.

    ``True`` means the value was **not** listed and was instead inferred.

    Attributes
    ----------
    materials : np.ndarray
        Boolean array of shape ``(num_materials,)``.
    category : bool
    price : bool
    mass : bool
    ratings : bool
    """

    materials: np.ndarray
    category: bool
    price: bool
    mass: bool
    ratings: bool