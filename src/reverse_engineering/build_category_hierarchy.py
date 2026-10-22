"""Extract the category hierarchy encoded in the EMMa records.

Each record stores its root-to-leaf category path in ``record["category"]``.
Category-search keys are ``(category_id, depth)`` pairs, so the same identity
is retained here (an ID alone is not necessarily sufficient).

The output pickle is a dictionary containing only standard Python objects:

``children``
    Direct child keys for every observed category (leaves map to ``()``).
``parents``
    Direct parent keys for every observed category (roots map to ``()``).
``roots``
    All observed root keys.
``subtree``
    Each key and all of its descendants, including itself.  Intersect this
    tuple with the keys of ``category_search_file.pkl`` to collect all search
    buckets below a selected category.
``occurrences``
    Number of dataset paths in which each category occurs.

Usage
-----
python build_category_hierarchy.py /path/to/emma_root
python build_category_hierarchy.py /path/to/emma_root -o hierarchy.pkl
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from freeze_dried_data.freeze_dried_data_old import FDD


CategoryKey = tuple[int, int]

DEFAULT_DATASET_FDD = "EMMa_v1.1.fdd"
DEFAULT_CATEGORY_SEARCH_PKL = "category_search_file.pkl"
DEFAULT_OUTPUT_PKL = "category_hierarchy.pkl"


def _category_path(record: dict[str, Any], asin: str) -> list[CategoryKey]:
    """Return one record's non-padding ``(id, depth)`` category path."""
    raw_path = record.get("category", ())
    try:
        values = list(raw_path)
    except TypeError as exc:
        raise TypeError(
            f"Record {asin!r} has a non-iterable category path: {raw_path!r}"
        ) from exc

    path: list[CategoryKey] = []
    padding_started = False
    for depth, raw_id in enumerate(values):
        try:
            category_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record {asin!r} has an invalid category ID at depth "
                f"{depth}: {raw_id!r}"
            ) from exc

        if category_id < 0:
            padding_started = True
        elif padding_started:
            raise ValueError(
                f"Record {asin!r} has category ID {category_id} after "
                "category padding began."
            )
        else:
            path.append((category_id, depth))
    return path


def _descendant_closure(
    children: dict[CategoryKey, tuple[CategoryKey, ...]],
) -> dict[CategoryKey, tuple[CategoryKey, ...]]:
    """Build inclusive descendant tuples, detecting malformed cycles."""
    memo: dict[CategoryKey, frozenset[CategoryKey]] = {}
    visiting: set[CategoryKey] = set()

    def visit(node: CategoryKey) -> frozenset[CategoryKey]:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError(f"Cycle detected in category hierarchy at {node!r}.")

        visiting.add(node)
        descendants = {node}
        for child in children[node]:
            descendants.update(visit(child))
        visiting.remove(node)
        memo[node] = frozenset(descendants)
        return memo[node]

    for key in children:
        visit(key)
    return {key: tuple(sorted(values)) for key, values in memo.items()}


def build_category_hierarchy(
    records: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a complete hierarchy from ``(asin, record)`` pairs."""
    children_sets: defaultdict[CategoryKey, set[CategoryKey]] = defaultdict(set)
    parent_sets: defaultdict[CategoryKey, set[CategoryKey]] = defaultdict(set)
    occurrences: defaultdict[CategoryKey, int] = defaultdict(int)
    path_count = 0

    for asin, record in records:
        path = _category_path(record, asin)
        path_count += 1
        for key in path:
            # Touch both maps so roots and leaves are explicitly represented.
            children_sets[key]
            parent_sets[key]
            occurrences[key] += 1
        for parent, child in zip(path, path[1:]):
            children_sets[parent].add(child)
            parent_sets[child].add(parent)

    children = {
        key: tuple(sorted(values)) for key, values in children_sets.items()
    }
    parents = {
        key: tuple(sorted(values)) for key, values in parent_sets.items()
    }
    roots = tuple(sorted(key for key in children if not parents[key]))

    return {
        "format_version": 1,
        "children": children,
        "parents": parents,
        "roots": roots,
        "subtree": _descendant_closure(children),
        "occurrences": dict(occurrences),
        "record_count": path_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract EMMa's category hierarchy from stored paths.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing EMMa_v1.1.fdd and category_search_file.pkl.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output pickle path (defaults to <root>/category_hierarchy.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = args.root / DEFAULT_DATASET_FDD
    category_search_path = args.root / DEFAULT_CATEGORY_SEARCH_PKL
    output = args.output or (args.root / DEFAULT_OUTPUT_PKL)

    with FDD(str(dataset_path), read_only=True) as dataset:
        hierarchy = build_category_hierarchy(dataset.items())

    with category_search_path.open("rb") as fh:
        category_search = pickle.load(fh)

    hierarchy_keys = set(hierarchy["children"])
    search_keys = set(category_search)
    missing = search_keys - hierarchy_keys
    if missing:
        sample = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing)} category-search keys were not found in the "
            f"dataset paths; first keys: {sample!r}"
        )

    hierarchy["category_search_keys"] = tuple(sorted(search_keys))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(hierarchy, fh, protocol=pickle.HIGHEST_PROTOCOL)

    edge_count = sum(len(values) for values in hierarchy["children"].values())
    print(f"Wrote {output}")
    print(
        f"records={hierarchy['record_count']:,}, "
        f"categories={len(hierarchy_keys):,}, edges={edge_count:,}, "
        f"roots={len(hierarchy['roots']):,}"
    )


if __name__ == "__main__":
    main()
