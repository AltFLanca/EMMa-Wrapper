#!/usr/bin/env python3
"""Convert an HDF5ArraySource file to an HDF5Source-compatible file.

The input stores values as dense arrays under ``data/<field>`` and uses an
ASIN index plus a presence mask. The output stores each present value as a
root-level dataset named after its ASIN, which is the layout expected by
``HDF5Source``. Multiple input fields are packed into a one-element structured
array per ASIN; use ``--field`` to extract one field as a plain array instead.

By default, the selected input field's complete compression configuration is
reused, including third-party filters registered by ``hdf5plugin``.

This is deliberately a standalone utility and is not part of the ``emma``
package.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

# Importing hdf5plugin registers its bundled filters with HDF5. It remains an
# optional dependency: files using only built-in HDF5 filters do not need it.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HDF5ArraySource file")
    parser.add_argument("output", type=Path, help="HDF5Source file to create")
    parser.add_argument(
        "--field",
        help=(
            "convert only this field; by default all fields are preserved in a "
            "one-element structured record"
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("input", "lzf", "gzip", "none"),
        default="input",
        help="output compression (default: copy from the input field)",
    )
    parser.add_argument(
        "--compression-field",
        help=(
            "field whose compression is copied when packing multiple fields "
            "(default: pred if present, otherwise the first field)"
        ),
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        choices=range(10),
        metavar="0-9",
        default=4,
        help="gzip level when --compression=gzip (default: 4)",
    )
    shuffle = parser.add_mutually_exclusive_group()
    shuffle.add_argument(
        "--shuffle",
        action="store_true",
        dest="shuffle",
        help="enable the HDF5 shuffle filter",
    )
    shuffle.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="disable the HDF5 shuffle filter",
    )
    checksum = parser.add_mutually_exclusive_group()
    checksum.add_argument(
        "--fletcher32",
        action="store_true",
        dest="fletcher32",
        help="enable Fletcher32 checksums",
    )
    checksum.add_argument(
        "--no-fletcher32",
        action="store_false",
        dest="fletcher32",
        help="disable Fletcher32 checksums",
    )
    parser.set_defaults(shuffle=None, fletcher32=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the completion summary",
    )
    return parser


def _read_layout(
    source: h5py.File,
    requested_field: str | None,
) -> tuple[Any, Any, dict[str, h5py.Dataset]]:
    try:
        asins = source["_index/asins"]
        present = source["_index/present"]
        data = source["data"]
    except KeyError as exc:
        raise ValueError(
            "input is not an HDF5ArraySource: expected _index/asins, "
            "_index/present, and data",
        ) from exc

    if asins.ndim != 1 or present.ndim != 1 or len(asins) != len(present):
        raise ValueError("invalid HDF5ArraySource index or presence-mask shape")

    fields = sorted(data.keys())
    if not fields:
        raise ValueError("input HDF5ArraySource has no fields")
    if requested_field is not None and requested_field not in data:
        choices = ", ".join(repr(name) for name in fields)
        raise ValueError(
            f"field {requested_field!r} does not exist; available fields: {choices}",
        )

    selected = fields if requested_field is None else [requested_field]
    values = {name: data[name] for name in selected}
    for name, dataset in values.items():
        if dataset.ndim < 1 or dataset.shape[0] != len(asins):
            raise ValueError(
                f"field {name!r} has shape {dataset.shape}, but its first "
                f"dimension must contain {len(asins)} rows",
            )
    return asins, present, values


def _custom_compression(dataset: h5py.Dataset) -> tuple[int, tuple[int, ...]]:
    """Return the single non-standard compression filter and its options."""
    excluded = {
        h5py.h5z.FILTER_DEFLATE,
        h5py.h5z.FILTER_SHUFFLE,
        h5py.h5z.FILTER_FLETCHER32,
        h5py.h5z.FILTER_SZIP,
        h5py.h5z.FILTER_NBIT,
        h5py.h5z.FILTER_SCALEOFFSET,
    }
    creation = dataset.id.get_create_plist()
    custom = []
    for index in range(creation.get_nfilters()):
        filter_id, _flags, options, _name = creation.get_filter(index)
        if filter_id not in excluded and filter_id != h5py.h5z.FILTER_LZF:
            custom.append((filter_id, options))
    if len(custom) != 1:
        raise ValueError(
            f"cannot identify a single custom compression filter on {dataset.name!r}",
        )
    return custom[0]


def _inherited_compression(dataset: h5py.Dataset) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if dataset.compression in {"gzip", "lzf", "szip"}:
        options["compression"] = dataset.compression
        if dataset.compression_opts is not None:
            options["compression_opts"] = dataset.compression_opts
    elif dataset.compression is not None:
        filter_id, filter_options = _custom_compression(dataset)
        if not h5py.h5z.filter_avail(filter_id):
            package_hint = (
                " Install hdf5plugin in this environment."
                if hdf5plugin is None
                else " Ensure the package providing that HDF5 filter is installed."
            )
            raise ValueError(
                f"HDF5 filter {filter_id} used by {dataset.name!r} is unavailable."
                + package_hint,
            )
        options["compression"] = filter_id
        options["compression_opts"] = filter_options
    return options


def _dataset_options(
    args: argparse.Namespace,
    value: np.ndarray,
    compression_source: h5py.Dataset,
) -> dict[str, Any]:
    if value.ndim == 0:
        return {}

    if args.compression == "input":
        options = _inherited_compression(compression_source)
    else:
        options = {}
    if args.compression in {"lzf", "gzip"}:
        options["compression"] = args.compression
        if args.compression == "gzip":
            options["compression_opts"] = args.gzip_level
    shuffle = compression_source.shuffle if args.shuffle is None else args.shuffle
    fletcher32 = (
        compression_source.fletcher32
        if args.fletcher32 is None
        else args.fletcher32
    )
    if shuffle:
        options["shuffle"] = True
    if fletcher32:
        options["fletcher32"] = True
    return options


def _row_value(values: dict[str, h5py.Dataset], index: int) -> np.ndarray:
    if len(values) == 1:
        return np.asarray(next(iter(values.values()))[index])

    dtype_fields = [
        (name, dataset.dtype, dataset.shape[1:])
        for name, dataset in values.items()
    ]
    record = np.empty(1, dtype=np.dtype(dtype_fields))
    for name, dataset in values.items():
        record[name][0] = dataset[index]
    return record


def _compression_source(
    args: argparse.Namespace,
    values: dict[str, h5py.Dataset],
) -> h5py.Dataset:
    if args.compression_field is not None:
        if args.compression_field not in values:
            choices = ", ".join(repr(name) for name in values)
            raise ValueError(
                f"compression field {args.compression_field!r} is not selected; "
                f"selected fields: {choices}",
            )
        return values[args.compression_field]
    if "pred" in values:
        return values["pred"]
    return next(iter(values.values()))


def convert(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if input_path == output_path:
        raise ValueError("input and output must be different files")
    if not input_path.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise ValueError(
            f"output already exists: {output_path} (use --overwrite to replace it)",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    converted = 0
    skipped = 0

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        with h5py.File(input_path, "r") as source, h5py.File(temporary_path, "w") as target:
            asins, present, values = _read_layout(source, args.field)
            compression_source = _compression_source(args, values)
            seen: set[str] = set()

            for index, is_present in enumerate(present):
                if not bool(is_present):
                    skipped += 1
                    continue

                raw_asin = asins[index]
                asin = raw_asin.decode("utf-8") if isinstance(raw_asin, bytes) else str(raw_asin)
                if not asin:
                    raise ValueError(f"ASIN at row {index} is empty")
                if "/" in asin:
                    raise ValueError(f"ASIN {asin!r} contains '/', which cannot be a dataset name")
                if asin in seen:
                    raise ValueError(f"duplicate present ASIN in input: {asin!r}")
                seen.add(asin)

                value = _row_value(values, index)
                target.create_dataset(
                    asin,
                    data=value,
                    dtype=value.dtype,
                    **_dataset_options(args, value, compression_source),
                )
                converted += 1

            target.flush()

        if args.overwrite:
            os.replace(temporary_path, output_path)
        else:
            # Unlike rename/replace, link fails atomically if another process
            # creates the destination while conversion is in progress.
            os.link(temporary_path, output_path)
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if not args.quiet:
        print(f"Converted {converted} datasets to {output_path} ({skipped} absent rows skipped).")
    return converted


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        convert(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
