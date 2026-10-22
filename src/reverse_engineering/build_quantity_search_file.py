"""Build ``quantity_search_file.pkl`` for the EMMa dataset.

This is intentionally separate from the package runtime code.
It mirrors the existing search-file convention:

- rows follow the canonical ASIN order from ``search_file.fdd``
- output is a 1-D NumPy array saved via pickle

Usage
-----
python build_quantity_search_file.py /path/to/emma_root
python build_quantity_search_file.py /path/to/emma_root -o custom_name.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from freeze_dried_data.freeze_dried_data_old import FDD


DEFAULT_DATASET_FDD = "EMMa_v1.1.fdd"
DEFAULT_SEARCH_FDD = "search_file.fdd"
DEFAULT_OUTPUT_PKL = "quantity_search_file.pkl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build quantity_search_file.pkl for EMMa.",
    )
    p.add_argument(
        "root",
        type=Path,
        help="Directory containing EMMa_v1.1.fdd and search_file.fdd.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output pickle path. Defaults to <root>/quantity_search_file.pkl."
        ),
    )
    return p.parse_args()


def _get_quantity(rec: dict, asin: str) -> int:
    try:
        value = rec["quantity"]
    except KeyError as exc:
        raise KeyError(f"Record {asin!r} is missing the 'quantity' field.") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Record {asin!r} has a non-integer quantity value: {value!r}"
        ) from exc


def build_quantity_search(root: Path) -> np.ndarray:
    dataset = FDD(str(root / DEFAULT_DATASET_FDD))
    search = FDD(str(root / DEFAULT_SEARCH_FDD))

    asins = list(search.keys())
    quantity = np.fromiter(
        (_get_quantity(dataset[asin], asin) for asin in asins),
        dtype=np.int64,
        count=len(asins),
    )
    return quantity


def main() -> None:
    args = _parse_args()
    root = args.root
    output = args.output or (root / DEFAULT_OUTPUT_PKL)

    quantity = build_quantity_search(root)

    with output.open("wb") as f:
        pickle.dump(quantity, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {output}")
    print(f"shape={quantity.shape}, dtype={quantity.dtype}")


if __name__ == "__main__":
    main()