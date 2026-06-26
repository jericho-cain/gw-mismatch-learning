from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.evaluation.geometry import distance_correlations
from gw_mismatch_learning.utils.io import load_arrays_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="HDF5 file with latent_distance and mismatch arrays.",
    )
    args = parser.parse_args()
    arrays = load_arrays_hdf5(args.input)
    report = distance_correlations(arrays["latent_distance"], arrays["mismatch"])
    for key, value in report.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
