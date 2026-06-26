from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.utils.io import save_arrays_hdf5
from gw_mismatch_learning.waveforms.banks import sample_binary_mass_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/waveform_bank.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    bank_cfg = config["bank"]
    bank = sample_binary_mass_bank(
        num_waveforms=int(bank_cfg["num_waveforms"]),
        mass_1_range=tuple(bank_cfg["mass_1_range"]),
        mass_2_range=tuple(bank_cfg["mass_2_range"]),
        seed=int(config.get("seed", 1234)),
    )
    save_arrays_hdf5(config["output"]["path"], {"mass_1": bank.mass_1, "mass_2": bank.mass_2})


if __name__ == "__main__":
    main()
