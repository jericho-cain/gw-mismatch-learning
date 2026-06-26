from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gw_mismatch_learning.datasets.synthetic import validate_distance_matrix
from gw_mismatch_learning.utils.io import (
    ensure_dir,
    load_arrays_hdf5,
    load_hdf5_attrs,
    save_arrays_hdf5,
)

REQUIRED_GW_METADATA = {
    "approximant",
    "mass_1_min",
    "mass_1_max",
    "mass_2_min",
    "mass_2_max",
    "spin_model",
    "sample_rate",
    "delta_f",
    "f_lower",
    "f_final",
    "psd",
    "num_waveforms",
    "pycbc_version",
    "lalsuite_version",
    "seed",
    "created_at_utc",
}


@dataclass(frozen=True)
class DistanceMatrixDataset:
    """Feature vectors paired with a target distance matrix."""

    features: np.ndarray
    distance: np.ndarray
    metadata: dict[str, Any]


def load_distance_matrix_dataset(path: str | Path) -> DistanceMatrixDataset:
    arrays = load_arrays_hdf5(path)
    metadata = {key: value for key, value in arrays.items() if key not in {"features", "distance"}}
    metadata.update(load_hdf5_attrs(path))
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
    attrs = {
        key: value
        for key, value in dataset.metadata.items()
        if isinstance(value, str | int | float | np.integer | np.floating)
    }
    save_arrays_hdf5(output_path, arrays, attrs=attrs)


def load_or_create_gw_mismatch_dataset(config: dict[str, Any], seed: int) -> DistanceMatrixDataset:
    from gw_mismatch_learning.waveforms.tiny_bank import build_tiny_waveform_mismatch_dataset

    cache_path = Path(config["cache_path"])
    if cache_path.exists() and not bool(config.get("overwrite", False)):
        dataset = load_distance_matrix_dataset(cache_path)
        if _has_required_metadata(dataset.metadata) or not _can_regenerate_dataset(config):
            return dataset

    dataset = build_tiny_waveform_mismatch_dataset(
        num_waveforms=int(config.get("num_waveforms", 32)),
        mass_1_range=tuple(config.get("mass_1_range", [20.0, 40.0])),
        mass_2_range=tuple(config.get("mass_2_range", [10.0, 30.0])),
        approximant=str(config.get("approximant", "IMRPhenomD")),
        delta_f=float(config.get("delta_f", 1.0 / 16.0)),
        f_lower=float(config.get("f_lower", 20.0)),
        f_final=float(config.get("f_final", 512.0)),
        psd_name=str(config.get("psd", "aLIGOZeroDetHighPower")),
        max_estimated_runtime_minutes=float(config.get("max_estimated_runtime_minutes", 30.0)),
        benchmark_pairs=int(config.get("benchmark_pairs", 128)),
        show_progress=bool(config.get("show_progress", True)),
        seed=seed,
    )
    save_distance_matrix_dataset(cache_path, dataset)
    return dataset


def _has_required_metadata(metadata: dict[str, Any]) -> bool:
    return REQUIRED_GW_METADATA.issubset(metadata)


def _can_regenerate_dataset(config: dict[str, Any]) -> bool:
    return "num_waveforms" in config
