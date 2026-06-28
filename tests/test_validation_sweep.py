from pathlib import Path

import yaml
from scripts.run_validation_sweep import build_runs, execute_runs, summarize_records, write_summary


def test_build_runs_can_select_seed_study() -> None:
    config = {
        "base_config": "configs/waveform_bank_small.yaml",
        "studies": {
            "seed": {"enabled": True, "seeds": [101, 202]},
            "latent_dim": {"enabled": True, "values": [2, 4]},
        },
    }
    runs = build_runs(config, selected_studies={"seed"})
    assert [run["run_id"] for run in runs] == ["seed_101", "seed_202"]
    assert {run["config"]["seed"] for run in runs} == {101, 202}
    assert {run["config"].get("distance_target", "mismatch") for run in runs} == {"mismatch"}


def test_build_runs_preserves_chordal_distance_target() -> None:
    config = {
        "base_config": "configs/waveform_bank_small.yaml",
        "distance_target": "chordal",
        "studies": {
            "seed": {"enabled": True, "seeds": [101]},
        },
    }

    runs = build_runs(config, selected_studies={"seed"})

    assert runs[0]["config"]["distance_target"] == "chordal"


def test_summarize_records_reports_mean_and_std() -> None:
    summary = summarize_records(
        [
            {"study": "seed", "run_id": "a", "spearman": 0.5},
            {"study": "seed", "run_id": "b", "spearman": 0.7},
        ]
    )
    assert summary["seed"]["spearman"]["mean"] == 0.6
    assert summary["seed"]["spearman"]["std"] > 0.0


def test_execute_runs_writes_structured_outputs(tmp_path: Path) -> None:
    base_config = {
        "seed": 1,
        "synthetic_data": {
            "num_samples": 16,
            "input_dim": 8,
            "metric": "cosine",
            "noise": 0.0,
        },
        "pairs": {"num_pairs": 32},
        "model": {"input_dim": 8, "embedding_dim": 2, "hidden_dims": [8]},
        "training": {"epochs": 1, "batch_size": 16, "learning_rate": 0.001},
        "evaluation": {"top_k": 2},
    }
    runs = [
        {
            "study": "seed",
            "run_id": "seed_1",
            "config": base_config,
            "parameters": {"seed": 1},
        }
    ]
    records = execute_runs(runs, tmp_path)
    assert records[0]["runtime_seconds"] >= 0.0
    assert records[0]["distance_target"] == "mismatch"
    assert (tmp_path / "seed" / "seed_1" / "config.yaml").exists()
    metrics = yaml.safe_load(
        (tmp_path / "seed" / "seed_1" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["distance_target"] == "mismatch"


def test_write_summary_records_distance_target_metadata(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    write_summary(
        path,
        [
            {
                "study": "seed",
                "run_id": "seed_1",
                "distance_target": "chordal",
                "spearman": 0.5,
            }
        ],
    )

    summary = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert summary["distance_targets"] == ["chordal"]
    assert summary["studies"]["seed"]["spearman"]["mean"] == 0.5


def test_phase3_config_loads() -> None:
    config = yaml.safe_load(Path("configs/phase3_validation.yaml").read_text(encoding="utf-8"))
    assert config["studies"]["seed"]["seeds"] == [101, 202, 303, 404, 505]
