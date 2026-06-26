from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_arrays_hdf5(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        for name, array in arrays.items():
            handle.create_dataset(name, data=array)
        if attrs:
            for key, value in attrs.items():
                handle.attrs[key] = value


def load_arrays_hdf5(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: dataset[()] for name, dataset in handle.items()}
