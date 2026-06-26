from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMassBank:
    mass_1: np.ndarray
    mass_2: np.ndarray


def sample_binary_mass_bank(
    num_waveforms: int,
    mass_1_range: tuple[float, float],
    mass_2_range: tuple[float, float],
    seed: int | None = None,
) -> BinaryMassBank:
    rng = np.random.default_rng(seed)
    mass_1 = rng.uniform(*mass_1_range, size=num_waveforms)
    mass_2 = rng.uniform(*mass_2_range, size=num_waveforms)
    swap = mass_2 > mass_1
    mass_1[swap], mass_2[swap] = mass_2[swap], mass_1[swap]
    return BinaryMassBank(mass_1=mass_1.astype(np.float32), mass_2=mass_2.astype(np.float32))
