from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import numpy as np
from tqdm import tqdm

from gw_mismatch_learning.datasets.synthetic import validate_distance_matrix
from gw_mismatch_learning.waveforms.banks import BinaryMassBank, sample_binary_mass_bank
from gw_mismatch_learning.waveforms.generate import generate_pycbc_fd_waveform


def build_tiny_waveform_mismatch_dataset(
    num_waveforms: int = 32,
    mass_1_range: tuple[float, float] = (20.0, 40.0),
    mass_2_range: tuple[float, float] = (10.0, 30.0),
    approximant: str = "IMRPhenomD",
    delta_f: float = 1.0 / 16.0,
    f_lower: float = 20.0,
    f_final: float = 512.0,
    psd_name: str = "aLIGOZeroDetHighPower",
    max_estimated_runtime_minutes: float = 30.0,
    benchmark_pairs: int = 128,
    show_progress: bool = True,
    seed: int | None = None,
):
    from gw_mismatch_learning.datasets.gw import DistanceMatrixDataset

    bank = sample_binary_mass_bank(
        num_waveforms=num_waveforms,
        mass_1_range=mass_1_range,
        mass_2_range=mass_2_range,
        seed=seed,
    )
    if show_progress:
        print("Generating waveform bank...")
    waveforms = []
    iterator = zip(bank.mass_1, bank.mass_2, strict=True)
    progress = tqdm(
        iterator,
        total=num_waveforms,
        unit="waveform",
        disable=not show_progress,
    )
    for mass_1, mass_2 in progress:
        waveforms.append(
            generate_pycbc_fd_waveform(
                mass1=float(mass_1),
                mass2=float(mass_2),
                approximant=approximant,
                delta_f=delta_f,
                f_lower=f_lower,
                f_final=f_final,
            )
        )
    mismatch = compute_pycbc_mismatch_matrix(
        waveforms=waveforms,
        delta_f=delta_f,
        f_lower=f_lower,
        psd_name=psd_name,
        max_estimated_runtime_minutes=max_estimated_runtime_minutes,
        benchmark_pairs=benchmark_pairs,
        show_progress=show_progress,
    )
    features = mass_feature_matrix(bank)
    return DistanceMatrixDataset(
        features=features,
        distance=mismatch,
        metadata={
            "mass_1": bank.mass_1,
            "mass_2": bank.mass_2,
            "approximant": approximant,
            "mass_1_min": float(mass_1_range[0]),
            "mass_1_max": float(mass_1_range[1]),
            "mass_2_min": float(mass_2_range[0]),
            "mass_2_max": float(mass_2_range[1]),
            "spin_model": "nonspinning",
            "sample_rate": "not_applicable_frequency_domain",
            "delta_f": float(delta_f),
            "f_lower": float(f_lower),
            "f_final": float(f_final),
            "psd": psd_name,
            "num_waveforms": int(num_waveforms),
            "pairwise_overlap_count": int(num_waveforms * (num_waveforms - 1) // 2),
            "max_estimated_runtime_minutes": float(max_estimated_runtime_minutes),
            "pycbc_version": _package_version("pycbc"),
            "lalsuite_version": _package_version("lalsuite"),
            "seed": -1 if seed is None else int(seed),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def mass_feature_matrix(bank: BinaryMassBank) -> np.ndarray:
    mass_1 = bank.mass_1.astype(np.float32)
    mass_2 = bank.mass_2.astype(np.float32)
    total_mass = mass_1 + mass_2
    symmetric_mass_ratio = (mass_1 * mass_2) / np.square(total_mass)
    chirp_mass = np.power(mass_1 * mass_2, 3.0 / 5.0) / np.power(total_mass, 1.0 / 5.0)
    features = np.column_stack([mass_1, mass_2, total_mass, symmetric_mass_ratio, chirp_mass])
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True).clip(min=1e-6)
    return ((features - mean) / std).astype(np.float32)


def compute_pycbc_mismatch_matrix(
    waveforms: Sequence,
    delta_f: float,
    f_lower: float,
    psd_name: str = "aLIGOZeroDetHighPower",
    max_estimated_runtime_minutes: float = 30.0,
    benchmark_pairs: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    try:
        from pycbc.filter import match
        from pycbc.psd import aLIGOZeroDetHighPower
    except ImportError as exc:
        raise ImportError(
            "pycbc is required to compute waveform mismatch. Install with `pip install -e .[gw]`."
        ) from exc

    if psd_name != "aLIGOZeroDetHighPower":
        raise ValueError(f"Unsupported PSD for tiny prototype: {psd_name}")

    n_waveforms = len(waveforms)
    max_len = max(len(waveform) for waveform in waveforms)
    prepared = []
    for waveform in waveforms:
        waveform = waveform.copy()
        waveform.resize(max_len)
        prepared.append(waveform)

    psd = aLIGOZeroDetHighPower(max_len, delta_f, f_lower)
    mismatch = np.zeros((n_waveforms, n_waveforms), dtype=np.float32)
    pairs = [(i, j) for i in range(n_waveforms) for j in range(i + 1, n_waveforms)]
    total_pairs = len(pairs)
    if show_progress:
        print(f"Pairwise overlap computations required: {total_pairs:,}")

    benchmark_count = min(max(1, benchmark_pairs), total_pairs)
    benchmark_start = perf_counter()
    for i, j in pairs[:benchmark_count]:
        mismatch_value = _pycbc_mismatch(
            match,
            prepared[i],
            prepared[j],
            psd=psd,
            f_lower=f_lower,
        )
        mismatch[i, j] = mismatch_value
        mismatch[j, i] = mismatch_value
    benchmark_elapsed = perf_counter() - benchmark_start
    seconds_per_pair = benchmark_elapsed / benchmark_count
    estimated_seconds = seconds_per_pair * total_pairs
    estimated_minutes = estimated_seconds / 60.0
    if show_progress:
        print(f"Benchmark pairs: {benchmark_count:,}")
        print(f"Estimated mismatch runtime: {estimated_minutes:.1f} minutes")

    if estimated_minutes > max_estimated_runtime_minutes:
        raise RuntimeError(
            "Estimated mismatch runtime "
            f"({estimated_minutes:.1f} minutes) exceeds configured limit "
            f"({max_estimated_runtime_minutes:.1f} minutes). "
            f"Required pairwise overlaps: {total_pairs:,}."
        )

    if show_progress:
        print("Computing mismatch matrix...")
    progress = tqdm(
        pairs[benchmark_count:],
        initial=benchmark_count,
        total=total_pairs,
        unit="pair",
        disable=not show_progress,
    )
    for i, j in progress:
        mismatch_value = _pycbc_mismatch(
            match,
            prepared[i],
            prepared[j],
            psd=psd,
            f_lower=f_lower,
        )
        mismatch[i, j] = mismatch_value
        mismatch[j, i] = mismatch_value

    mismatch = np.clip(mismatch, 0.0, None)
    validate_distance_matrix(mismatch)
    return mismatch


def _pycbc_mismatch(match_fn, left, right, psd, f_lower: float) -> float:
    value, _ = match_fn(left, right, psd=psd, low_frequency_cutoff=f_lower)
    return 1.0 - float(value)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"
