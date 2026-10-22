"""Build a presence mask for package dimensions.

This file is intentionally external to the package, mirroring the other
search-file builders.  It produces a boolean ``.npy`` file that can be
loaded by :class:`EmmaDataset` and queried via
``SearchQuery(require_present=["package_dimensions"])``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from emma import EmmaDataset
from emma import _constants as C


def has_package_dimensions(rec: dict) -> bool:
    """Return ``True`` when the record contains package dimensions.

    This is a *presence* test, not a validity or EM-status test.
    """
    value = rec.get(C.RecordField.PACKAGE_DIMENSIONS)
    if value is None:
        return False
    arr = np.asarray(value)
    return arr.size > 0 and np.isfinite(arr).all()



def build_mask(root: Path) -> np.ndarray:
    ds = EmmaDataset(root)
    mask = np.fromiter(
        (has_package_dimensions(ds[asin]) for asin in ds.asins),
        dtype=bool,
        count=len(ds),
    )
    return mask



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing EMMa files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path (defaults to the canonical search-file name).",
    )
    args = parser.parse_args()

    output = args.output or (args.root / C.PACKAGE_DIMENSIONS_PRESENT_NPY)
    mask = build_mask(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, mask)

    print(f"Saved {mask.sum():,} / {mask.size:,} present flags to {output}")


if __name__ == "__main__":
    main()
