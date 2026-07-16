from pathlib import Path

import numpy as np
from scripts.run_principal_k8 import load_phase3_reference, validate_inputs


def test_phase3_reference_is_unique_per_query() -> None:
    reference = load_phase3_reference(Path("outputs/phase3_candidate_retrieval/query_results.csv"))
    assert reference["query_m1"].shape == (512,)
    assert reference["query_m2"].shape == (512,)
    assert reference["best_index"].shape == (512,)
    assert reference["best_match"].shape == (512,)
    assert np.all(reference["best_match"] > 0)


def test_principal_inputs_require_k8_original_seed() -> None:
    config = {
        "seed": 1234,
        "latent_dim": 8,
        "bank_sizes": [8192],
        "cache_template": "data/processed/waveform_bank_{bank_size}_mismatch.h5",
        "latent_sweep_reference": (
            "outputs/latent_dimension_sweep/runs/n8192_k8_seed1234/metrics.json"
        ),
        "phase3_config": "configs/phase3_candidate_retrieval.yaml",
    }
    validate_inputs(config)
