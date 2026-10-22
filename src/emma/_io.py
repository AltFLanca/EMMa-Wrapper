"""Low-level I/O helpers for JSON, pickle and NumPy files."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    """Deserialise a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_pickle(path: Path) -> Any:
    """Deserialise a single pickle file."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_pickle_required(
    path: Path,
    hint: str = "Run emma_completion.py first.",
) -> Any:
    """Load a pickle that *must* exist, or raise with a helpful message."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}. {hint}")
    return load_pickle(path)


def load_npy_optional(path: Path) -> np.ndarray | None:
    """Load a ``.npy`` file if present; return ``None`` otherwise."""
    return np.load(path) if path.exists() else None