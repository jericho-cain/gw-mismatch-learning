from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.evaluation.retrieval import retrieval_report
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.io import load_arrays_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="HDF5 file with embeddings and mismatch arrays.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    arrays = load_arrays_hdf5(args.input)
    index = SklearnNeighborIndex(arrays["embeddings"])
    _, retrieved = index.query(arrays["embeddings"], top_k=args.top_k)
    report = retrieval_report(arrays["mismatch"], retrieved, top_k=args.top_k)
    for key, value in report.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
