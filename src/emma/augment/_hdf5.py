"""HDF5-backed augmentation sources."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._base import AugmentSource


def _require_h5py(source_name: str) -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"h5py is required for {source_name}. "
            "Install with: pip install h5py",
        ) from exc
    return h5py


@dataclass(frozen=True, slots=True)
class HDF5FieldSpec:
    """Schema for a dense field stored inside :class:`HDF5ArraySource`.

    Parameters
    ----------
    shape : tuple[int, ...], optional
        Per-item shape. When omitted, the shape is inferred from the first
        saved value for the field.
    dtype : NumPy dtype-like, optional
        Stored dtype. When omitted, inferred from the first saved value.
    fillvalue : Any, optional
        HDF5 fill value used when allocating the dataset.
    chunks : tuple[int, ...] or bool, optional
        Explicit HDF5 chunk layout for the full dataset shape
        ``(n_items, *shape)``. When omitted, a row-wise chunk layout is used.
    """

    shape: tuple[int, ...] | None = None
    dtype: Any | None = None
    fillvalue: Any | None = None
    chunks: tuple[int, ...] | bool | None = None


class HDF5ArraySource(AugmentSource):
    """Dense HDF5-backed augmentation store with one row per ASIN.

    Designed for large, uniform-shape outputs such as depth maps,
    embeddings, and dense per-item tensors. The file stores a canonical
    ASIN index, a resumable ``present`` mask, and one or more field
    datasets whose first dimension is the dataset row.

    When exactly one field exists, ``source[asin]`` returns that array
    directly. Otherwise it returns a ``dict`` mapping field names to the
    stored values for that ASIN.
    """

    __slots__ = (
        "_name",
        "_path",
        "_file",
        "_index_group",
        "_data_group",
        "_asins",
        "_asin_to_index",
        "_present",
        "_field_specs",
        "_field_names",
        "_compression",
        "_compression_opts",
        "_shuffle",
        "_fletcher32",
    )

    def __init__(
        self,
        name: str,
        path: str | Path,
        mode: str = "r",
        *,
        asins: Sequence[str] | None = None,
        fields: Mapping[str, HDF5FieldSpec] | None = None,
        compression: str | None = "lzf",
        compression_opts: Any | None = None,
        shuffle: bool = True,
        fletcher32: bool = False,
    ) -> None:
        h5py = _require_h5py("HDF5ArraySource")

        self._name = name
        self._path = Path(path)
        self._compression = compression
        self._compression_opts = compression_opts
        self._shuffle = shuffle
        self._fletcher32 = fletcher32
        self._field_specs = dict(fields or {})

        if mode in {"a", "w", "w-", "x"}:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(str(self._path), mode)
        self._index_group = self._file.require_group("_index")
        self._data_group = self._file.require_group("data")

        if "asins" not in self._index_group:
            if asins is None:
                raise ValueError(
                    "'asins' must be provided when creating a new HDF5ArraySource.",
                )
            self._initialise_index(asins, h5py)
            self._file.attrs["emma_hdf5_array_source"] = True
            self._file.attrs["name"] = name
            self._file.attrs["field_names"] = json.dumps(list(self._field_specs))
        else:
            if asins is not None:
                self._validate_asins(asins)
            if "field_names" not in self._file.attrs and self._field_specs:
                self._file.attrs["field_names"] = json.dumps(list(self._field_specs))

        self._asins = self._index_group["asins"].asstr()[()]
        self._asin_to_index = {a: i for i, a in enumerate(self._asins.tolist())}
        self._present = self._index_group["present"]
        self._field_names = self._load_field_names()

    @property
    def name(self) -> str:
        return self._name

    @property
    def field_names(self) -> tuple[str, ...]:
        return self._field_names

    @property
    def asins(self) -> np.ndarray:
        return self._asins.copy()

    @property
    def present_mask(self) -> np.ndarray:
        return self._present[()].astype(bool, copy=False)

    def __getitem__(self, asin: str) -> dict[str, Any]:
        idx = self._row_index(asin)
        if not bool(self._present[idx]):
            raise KeyError(asin)
        return self._read_row(idx)

    def __contains__(self, asin: str) -> bool:
        idx = self._asin_to_index.get(asin)
        return False if idx is None else bool(self._present[idx])

    def __len__(self) -> int:
        return int(np.count_nonzero(self._present[()]))

    def keys(self) -> Iterable[str]:
        present = self._present[()].astype(bool, copy=False)
        return self._asins[present].tolist()

    def get_batch(self, asins: Sequence[str]) -> dict[str, Any]:
        return {a: self[a] for a in asins if a in self}

    def save(self, asin: str, data: Any, *, flush: bool = True) -> None:
        idx = self._row_index(asin)
        row = self._normalise_row_payload(data)
        self._ensure_field_datasets(row)
        self._validate_row_payload(row)
        for field_name, value in row.items():
            self._data_group[field_name][idx] = value
        self._present[idx] = True
        if flush:
            self._file.flush()

    def save_batch(self, batch: Mapping[str, Any], *, flush: bool = True) -> None:
        for asin, data in batch.items():
            self.save(asin, data, flush=False)
        if flush:
            self._file.flush()

    def delete(self, asin: str, *, flush: bool = True) -> None:
        idx = self._row_index(asin)
        self._present[idx] = False
        if flush:
            self._file.flush()

    def missing_keys(self) -> list[str]:
        present = self._present[()].astype(bool, copy=False)
        return self._asins[~present].tolist()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> HDF5ArraySource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._file.close()
        except Exception:  # noqa: BLE001
            pass

    def _initialise_index(self, asins: Sequence[str], h5py: Any) -> None:
        asins_arr = np.asarray(list(asins), dtype=object)
        n = len(asins_arr)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        self._index_group.create_dataset("asins", data=asins_arr, dtype=str_dtype)
        chunk_len = min(max(n, 1), 8192)
        self._index_group.create_dataset(
            "present",
            shape=(n,),
            dtype=np.bool_,
            chunks=(chunk_len,),
            fillvalue=False,
        )

    def _validate_asins(self, asins: Sequence[str]) -> None:
        existing = self._index_group["asins"].asstr()[()].tolist()
        candidate = list(asins)
        if existing != candidate:
            raise ValueError(
                "The provided ASIN ordering does not match the existing HDF5ArraySource.",
            )

    def _load_field_names(self) -> tuple[str, ...]:
        if len(self._data_group):
            return tuple(self._data_group.keys())
        raw = self._file.attrs.get("field_names")
        if raw is None:
            return tuple(self._field_specs)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return tuple(json.loads(raw))

    def _row_index(self, asin: str) -> int:
        try:
            return self._asin_to_index[asin]
        except KeyError:
            raise KeyError(asin) from None

    def _read_row(self, idx: int) -> dict[str, Any]:
        return {field: self._data_group[field][idx] for field in self._field_names}

    def _normalise_row_payload(self, data: Any) -> dict[str, Any]:
        if len(self._field_names) == 1 and not isinstance(data, Mapping):
            return {self._field_names[0]: data}
        if not isinstance(data, Mapping):
            raise TypeError(
                "Multi-field HDF5ArraySource.save() expects a mapping of field names to values.",
            )
        row = dict(data)
        missing = [name for name in self._field_names if name not in row]
        extra = [name for name in row if name not in self._field_names]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing fields: {missing}")
            if extra:
                problems.append(f"unexpected fields: {extra}")
            raise ValueError("Invalid row payload (" + "; ".join(problems) + ").")
        return row

    def _ensure_field_datasets(self, row: Mapping[str, Any]) -> None:
        for field_name, value in row.items():
            if field_name in self._data_group:
                continue
            spec = self._field_specs.get(field_name, HDF5FieldSpec())
            self._create_field_dataset(field_name, value, spec)
        self._field_names = self._load_field_names()

    def _create_field_dataset(
        self,
        field_name: str,
        value: Any,
        spec: HDF5FieldSpec,
    ) -> None:
        arr = np.asarray(value)
        item_shape = spec.shape if spec.shape is not None else arr.shape
        dtype = np.dtype(spec.dtype) if spec.dtype is not None else arr.dtype
        if tuple(arr.shape) != tuple(item_shape):
            raise ValueError(
                f"Field {field_name!r} expected item shape {item_shape}, got {arr.shape}.",
            )

        full_shape = (len(self._asins), *item_shape)
        chunks = spec.chunks
        if chunks is None:
            chunks = self._default_chunks(full_shape)
        kwargs = {
            "shape": full_shape,
            "dtype": dtype,
            "chunks": chunks,
        }
        if spec.fillvalue is not None:
            kwargs["fillvalue"] = spec.fillvalue
        if self._compression is not None:
            kwargs["compression"] = self._compression
        if self._compression_opts is not None:
            kwargs["compression_opts"] = self._compression_opts
        if self._shuffle and len(full_shape) > 0:
            kwargs["shuffle"] = True
        if self._fletcher32 and len(full_shape) > 0:
            kwargs["fletcher32"] = True

        self._data_group.create_dataset(field_name, **kwargs)

    @staticmethod
    def _default_chunks(full_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(full_shape) == 1:
            return (min(max(full_shape[0], 1), 8192),)
        return (1, *full_shape[1:])

    def _validate_row_payload(self, row: Mapping[str, Any]) -> None:
        for field_name, value in row.items():
            ds = self._data_group[field_name]
            arr = np.asarray(value)
            expected_shape = ds.shape[1:]
            if tuple(arr.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Field {field_name!r} expected item shape {expected_shape}, got {arr.shape}.",
                )


class HDF5Source(AugmentSource):
    """Single HDF5 file keyed by ASIN (requires ``h5py``).

    Suitable for flexible per-item datasets when you do not want a dense,
    pre-indexed store. Implements the context-manager protocol for
    deterministic cleanup.

    Parameters
    ----------
    name : str
        Augmentation name.
    path : str or Path
        HDF5 file path.
    mode : str
        ``"r"`` (read-only, default) or ``"a"`` (read/write, creates
        the file if absent).
    compression : str, optional
        HDF5 compression filter applied to newly-created datasets.
    compression_opts : Any, optional
        Compression-specific options.
    shuffle : bool
        Whether to enable the HDF5 shuffle filter for new datasets.
    """

    __slots__ = (
        "_name",
        "_path",
        "_file",
        "_compression",
        "_compression_opts",
        "_shuffle",
        "_fletcher32",
    )

    def __init__(
        self,
        name: str,
        path: str | Path,
        mode: str = "r",
        *,
        compression: str | None = "lzf",
        compression_opts: Any | None = None,
        shuffle: bool = True,
        fletcher32: bool = False,
    ) -> None:
        h5py = _require_h5py("HDF5Source")

        self._name = name
        self._path = Path(path)
        self._compression = compression
        self._compression_opts = compression_opts
        self._shuffle = shuffle
        self._fletcher32 = fletcher32
        if mode in {"a", "w", "w-", "x"}:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(str(self._path), mode)

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, asin: str) -> np.ndarray:
        if asin not in self._file:
            raise KeyError(asin)
        return self._file[asin][()]

    def __contains__(self, asin: str) -> bool:
        return asin in self._file

    def __len__(self) -> int:
        return len(self._file)

    def keys(self) -> Iterable[str]:
        return self._file.keys()

    def get_batch(self, asins: Sequence[str]) -> dict[str, np.ndarray]:
        """Batch read -- single file handle, no redundant membership checks."""
        f = self._file
        return {a: f[a][()] for a in asins if a in f}

    def save(self, asin: str, data: np.ndarray) -> None:
        """Write *data* as a dataset keyed by *asin* (overwrites)."""
        if asin in self._file:
            del self._file[asin]
        arr = np.asarray(data)
        kwargs: dict[str, Any] = {"data": arr}
        if arr.ndim > 0 and self._compression is not None:
            kwargs["compression"] = self._compression
        if arr.ndim > 0 and self._compression_opts is not None:
            kwargs["compression_opts"] = self._compression_opts
        if arr.ndim > 0 and self._shuffle:
            kwargs["shuffle"] = True
        if arr.ndim > 0 and self._fletcher32:
            kwargs["fletcher32"] = True
        self._file.create_dataset(asin, **kwargs)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> HDF5Source:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._file.close()
        except Exception:  # noqa: BLE001
            pass
