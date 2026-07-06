import numpy as np
from scripts.run_phase3_candidate_retrieval import (
    fit_feature_scaler,
    phase1_training_config,
    protect_output_dir,
    query_rows_for_k,
    raw_feature_matrix_from_masses,
    sample_query_bank,
    summarize_query_rows,
)

from gw_mismatch_learning.waveforms.banks import BinaryMassBank


def test_query_features_use_bank_scaler() -> None:
    bank_features = raw_feature_matrix_from_masses(
        np.asarray([30.0, 40.0], dtype=np.float32),
        np.asarray([20.0, 10.0], dtype=np.float32),
    )
    query_features = raw_feature_matrix_from_masses(
        np.asarray([35.0], dtype=np.float32),
        np.asarray([15.0], dtype=np.float32),
    )

    scaler = fit_feature_scaler(bank_features)
    transformed = scaler.transform(query_features)

    expected = (query_features - bank_features.mean(axis=0, keepdims=True)) / bank_features.std(
        axis=0,
        keepdims=True,
    ).clip(min=1e-6)
    np.testing.assert_allclose(transformed, expected.astype(np.float32))


def test_phase1_training_config_scales_pair_count_for_bank_size() -> None:
    base_config = {
        "seed": 1234,
        "gw_data": {"cache_path": "old.h5", "num_waveforms": 256, "overwrite": True},
        "pairs": {"num_pairs": 8192},
        "model": {"input_dim": 5, "embedding_dim": 4, "hidden_dims": [32, 16]},
        "training": {"epochs": 10, "batch_size": 128, "learning_rate": 0.001},
    }

    config = phase1_training_config(base_config, 8192, bank_cache_path="bank.h5")

    assert config["gw_data"]["num_waveforms"] == 8192
    assert config["gw_data"]["cache_path"] == "bank.h5"
    assert config["gw_data"]["overwrite"] is False
    assert config["pairs"]["num_pairs"] == 8192 * 32
    assert base_config["pairs"]["num_pairs"] == 8192


def test_protect_output_dir_blocks_existing_phase3_tables(tmp_path) -> None:
    output_dir = tmp_path / "phase3"
    output_dir.mkdir()
    (output_dir / "summary.csv").write_text("already here", encoding="utf-8")

    try:
        protect_output_dir(output_dir)
    except FileExistsError as exc:
        assert "allow_overwrite_output" in str(exc)
    else:
        raise AssertionError("Expected existing Phase III output files to be protected")

    protect_output_dir(output_dir, allow_overwrite=True)


def test_sample_query_bank_avoids_exact_bank_mass_duplicates() -> None:
    metadata = {
        "mass_1_min": 20.0,
        "mass_1_max": 40.0,
        "mass_2_min": 10.0,
        "mass_2_max": 30.0,
    }
    bank_masses = np.asarray([[30.0, 20.0]], dtype=np.float32)

    queries = sample_query_bank(
        num_queries=8,
        metadata=metadata,
        bank_masses=bank_masses,
        seed=9876,
    )

    bank_set = {(30.0, 20.0)}
    query_set = {
        (float(mass_1), float(mass_2))
        for mass_1, mass_2 in zip(queries.mass_1, queries.mass_2, strict=True)
    }
    assert query_set.isdisjoint(bank_set)
    assert len(queries.mass_1) == 8


def test_query_rows_and_summary_compute_delta_match_metrics() -> None:
    method = {
        "name": "learned_latent",
        "label": "Learned latent",
        "feature_names": ["learned_latent_coordinates"],
    }
    candidate_indices = np.asarray([[2, 1], [0, 3]], dtype=np.int64)
    candidate_matches = np.asarray([[0.97, 0.95], [0.80, 0.81]], dtype=np.float32)
    exhaustive_indices = np.asarray([2, 1], dtype=np.int64)
    exhaustive_matches = np.asarray([0.97, 0.90], dtype=np.float32)
    query_bank = BinaryMassBank(
        mass_1=np.asarray([32.0, 34.0], dtype=np.float32),
        mass_2=np.asarray([18.0, 16.0], dtype=np.float32),
    )

    rows = query_rows_for_k(
        method,
        2,
        candidate_indices,
        candidate_matches,
        exhaustive_indices,
        exhaustive_matches,
        query_bank,
    )
    summary = summarize_query_rows(method, 2, rows, bank_size=8)

    assert rows[0]["exact_best_recovered"] == 1
    assert rows[1]["exact_best_recovered"] == 0
    assert summary["candidate_fraction"] == 0.25
    assert summary["candidate_match_evaluations"] == 4
    assert summary["exhaustive_match_evaluations"] == 16
    assert summary["exact_best_recovery_rate"] == 0.5
    np.testing.assert_allclose(summary["delta_match_median"], 0.045, atol=1e-6)
