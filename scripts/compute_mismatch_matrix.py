from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.gw import load_or_create_gw_mismatch_dataset
from gw_mismatch_learning.datasets.waveform_dataset import make_mock_waveform_dataset
from gw_mismatch_learning.utils.io import save_arrays_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a mismatch-matrix dataset for training experiments."
    )
    parser.add_argument("--config", help="Config with a gw_data section.")
    parser.add_argument("--output", default="data/processed/mock_mismatch.h5")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        seed = int(config.get("seed", args.seed))
        dataset = load_or_create_gw_mismatch_dataset(config["gw_data"], seed=seed)
        print(f"features: {dataset.features.shape}")
        print(f"distance: {dataset.distance.shape}")
        print(f"cache: {config['gw_data']['cache_path']}")
        return

    data = make_mock_waveform_dataset(args.num_samples, args.input_dim, seed=args.seed)
    save_arrays_hdf5(args.output, {"features": data.features, "mismatch": data.mismatch})


if __name__ == "__main__":
    main()
