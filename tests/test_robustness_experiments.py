from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from scripts.run_robustness_experiments import (
    aggregate_records,
    build_specs,
    experiment_config,
    file_digest,
    student_t_summary,
    validate_existing_run,
    validate_specs,
)


def test_student_t_interval_uses_sample_std_and_df_four() -> None:
    mean, std, low, high = student_t_summary([1, 2, 3, 4, 5])
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.5811388300841898)
    assert low == pytest.approx(1.0367568385)
    assert high == pytest.approx(4.9632431615)


def test_latent_sweep_changes_only_run_scale_seed_and_latent_dimension() -> None:
    base = {
        "seed": 1234,
        "gw_data": {"cache_path": "old.h5", "num_waveforms": 10, "overwrite": False},
        "pairs": {"num_pairs": 8192},
        "model": {"input_dim": 5, "embedding_dim": 4, "hidden_dims": [32, 16]},
        "training": {"epochs": 10, "batch_size": 128, "learning_rate": 0.001},
        "evaluation": {"k_values": [5, 10, 20]},
        "outputs": {"save_metrics": True},
    }
    original = copy.deepcopy(base)
    result = experiment_config(
        base, {"seed": 2345, "bank_size": 8192, "latent_dim": 3, "cache_path": "bank.h5"}
    )
    assert base == original
    assert result["model"] == {"input_dim": 5, "embedding_dim": 3, "hidden_dims": [32, 16]}
    assert result["training"] == original["training"]
    assert result["evaluation"] == original["evaluation"]
    assert result["pairs"]["num_pairs"] == 8192 * 32
    assert result["gw_data"]["overwrite"] is False


def test_required_cache_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_specs(
            [
                {
                    "bank_size": 8,
                    "latent_dim": 4,
                    "seed": 1,
                    "cache_path": str(tmp_path / "missing.h5"),
                }
            ]
        )


def test_specs_are_five_seeds_per_requested_setting(tmp_path: Path) -> None:
    cache = tmp_path / "bank.h5"
    cache.touch()
    cfg = {
        "objective": "latent_dimension_sweep",
        "bank_size": 8192,
        "cache_path": str(cache),
        "latent_dimensions": [2, 4],
        "seeds": [1234, 2345, 3456, 4567, 5678],
    }
    specs = build_specs(cfg)
    validate_specs(specs)
    assert len(specs) == 10
    assert sum(spec["latent_dim"] == 4 for spec in specs) == 5


def test_aggregation_refuses_failed_or_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="failed"):
        aggregate_records([{"status": "failed", "latent_dim": 2}], "latent_dim")
    records = [{"status": "completed", "latent_dim": 2} for _ in range(4)]
    with pytest.raises(ValueError, match="exactly five"):
        aggregate_records(records, "latent_dim")


def test_resume_validates_config_cache_and_completion_without_mutation(tmp_path: Path) -> None:
    cache = tmp_path / "bank.h5"
    cache.write_bytes(b"frozen")
    spec = {"bank_size": 8, "latent_dim": 3, "seed": 1234, "cache_path": str(cache)}
    config = {"seed": 1234, "model": {"embedding_dim": 3}}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metrics = {
        **spec,
        "status": "completed",
        "cache_reused": True,
        "cache_sha256": file_digest(cache),
        "training_history": [0.2, 0.1],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    valid, reason, _ = validate_existing_run(run_dir, config, spec)
    assert valid is True
    assert reason == "valid"
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before

    metrics["seed"] = 999
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    valid, reason, _ = validate_existing_run(run_dir, config, spec)
    assert valid is False
    assert reason == "requested field mismatch: seed"
