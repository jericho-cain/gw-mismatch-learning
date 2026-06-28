from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.evaluation.metric_properties import (
    chordal_from_mismatch,
    metric_property_report,
)
from gw_mismatch_learning.utils.io import ensure_dir, load_arrays_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HDF5 file containing a mismatch matrix.")
    parser.add_argument(
        "--dataset",
        default="distance",
        help="Dataset name for the mismatch matrix.",
    )
    parser.add_argument("--mode", choices=["exhaustive", "sampled"], default="sampled")
    parser.add_argument("--num-triples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    report = run(
        input_path=Path(args.input),
        dataset=args.dataset,
        mode=args.mode,
        num_triples=args.num_triples,
        seed=args.seed,
        tol=args.tol,
    )

    if args.output:
        output_path = Path(args.output)
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print_summary(report)


def run(
    *,
    input_path: Path,
    dataset: str,
    mode: str,
    num_triples: int,
    seed: int,
    tol: float,
) -> dict[str, Any]:
    arrays = load_arrays_hdf5(input_path)
    if dataset not in arrays:
        available = ", ".join(sorted(arrays))
        raise KeyError(f"dataset '{dataset}' not found in {input_path}; available: {available}")

    mismatch = arrays[dataset]
    chordal = chordal_from_mismatch(mismatch)
    return {
        "input": str(input_path),
        "dataset": dataset,
        "mode": mode,
        "num_triples": int(num_triples),
        "seed": int(seed),
        "tol": float(tol),
        "mismatch": metric_property_report(
            mismatch,
            mode=mode,  # type: ignore[arg-type]
            num_triples=num_triples,
            seed=seed,
            tol=tol,
        ),
        "chordal": metric_property_report(
            chordal,
            mode=mode,  # type: ignore[arg-type]
            num_triples=num_triples,
            seed=seed,
            tol=tol,
        ),
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"Input: {report['input']}")
    print(f"Dataset: {report['dataset']}")
    for name in ["mismatch", "chordal"]:
        result = report[name]
        print(f"\n{name}:")
        print(f"  n: {result['n']}")
        print(f"  nonnegative: {result['nonnegative']} min={result['min_value']:.6g}")
        print(
            "  symmetric: "
            f"{result['symmetric']} max_error={result['max_symmetry_error']:.6g}"
        )
        print(
            "  zero_diagonal: "
            f"{result['zero_diagonal']} max_abs={result['max_diagonal_abs']:.6g}"
        )
        print(f"  offdiag near-zero count: {result['num_offdiag_near_zero']}")
        print(
            "  triangle violations: "
            f"{result['num_triangle_violations']} / {result['num_triangle_tests']} "
            f"({result['triangle_violation_fraction']:.6g})"
        )
        print(f"  max triangle violation: {result['max_triangle_violation']:.6g}")
        if result["example_violation"] is not None:
            print(f"  example violation: {result['example_violation']}")


if __name__ == "__main__":
    main()
