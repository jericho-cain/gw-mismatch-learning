from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gw_mismatch_learning.datasets.synthetic import validate_distance_matrix
from gw_mismatch_learning.utils.io import ensure_dir, load_arrays_hdf5, save_arrays_hdf5


@dataclass(frozen=True)
class DistanceMatrixDataset:
    """Feature vectors paired with a target distance matrix."""

    features: np.ndarray
    distance: np.ndarray
    metadata: dict[str, Any]


def load_distance_matrix_dataset(path: str | Path) -> DistanceMatrixDataset:
    arrays = load_arrays_hdf5(path)
    metadata = {key: value for key, value in arrays.items() if key not in {"features", "distance"}}
    dataset = DistanceMatrixDataset(
        features=arrays["features"].astype(np.float32),
        distance=arrays["distance"].astype(np.float32),
        metadata=metadata,
    )
    validate_distance_matrix(dataset.distance)
    return dataset


def save_distance_matrix_dataset(
    path: str | Path,
    dataset: DistanceMatrixDataset,
) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    arrays = {
        "features": dataset.features.astype(np.float32),
        "distance": dataset.distance.astype(np.float32),
    }
    for key, value in dataset.metadata.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
    save_arrays_hdf5(output_path, arrays)


def load_or_create_gw_mismatch_dataset(config: dict[str, Any], seed: int) -> DistanceMatrixDataset:
    from gw_mismatch_learning.waveforms.tiny_bank import build_tiny_waveform_mismatch_dataset

    cache_path = Path(config["cache_path"])
    if cache_path.exists() and not bool(config.get("overwrite", False)):
        return load_distance_matrix_dataset(cache_path)

    dataset = build_tiny_waveform_mismatch_dataset(
        num_waveforms=int(config.get("num_waveforms", 32)),
        mass_1_range=tuple(config.get("mass_1_range", [20.0, 40.0])),
        mass_2_range=tuple(config.get("mass_2_range", [10.0, 30.0])),
        approximant=str(config.get("approximant", "IMRPhenomD")),
        delta_f=float(config.get("delta_f", 1.0 / 16.0)),
        f_lower=float(config.get("f_lower", 20.0)),
        f_final=float(config.get("f_final", 512.0)),
        psd_name=str(config.get("psd", "aLIGOZeroDetHighPower")),
        seed=seed,
    )
    save_distance_matrix_dataset(cache_path, dataset)
    return dataset
