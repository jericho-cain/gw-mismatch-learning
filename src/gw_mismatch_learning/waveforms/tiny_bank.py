from __future__ import annotations

from collections.abc import Sequence

import numpy as np

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
    seed: int | None = None,
):
    from gw_mismatch_learning.datasets.gw import DistanceMatrixDataset

    bank = sample_binary_mass_bank(
        num_waveforms=num_waveforms,
        mass_1_range=mass_1_range,
        mass_2_range=mass_2_range,
        seed=seed,
    )
    waveforms = [
        generate_pycbc_fd_waveform(
            mass1=float(mass_1),
            mass2=float(mass_2),
            approximant=approximant,
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final,
        )
        for mass_1, mass_2 in zip(bank.mass_1, bank.mass_2, strict=True)
    ]
    mismatch = compute_pycbc_mismatch_matrix(
        waveforms=waveforms,
        delta_f=delta_f,
        f_lower=f_lower,
        psd_name=psd_name,
    )
    features = mass_feature_matrix(bank)
    return DistanceMatrixDataset(
        features=features,
        distance=mismatch,
        metadata={
            "mass_1": bank.mass_1,
            "mass_2": bank.mass_2,
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
    for i in range(n_waveforms):
        for j in range(i + 1, n_waveforms):
            value, _ = match(prepared[i], prepared[j], psd=psd, low_frequency_cutoff=f_lower)
            mismatch_value = 1.0 - float(value)
            mismatch[i, j] = mismatch_value
            mismatch[j, i] = mismatch_value

    mismatch = np.clip(mismatch, 0.0, None)
    validate_distance_matrix(mismatch)
    return mismatch
