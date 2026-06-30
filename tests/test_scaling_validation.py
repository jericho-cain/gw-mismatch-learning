from pathlib import Path

import yaml
from scripts.run_scaling_validation import (
    config_for_bank_size,
    failed_record,
    write_summary,
    write_summary_csv,
)


def test_config_for_bank_size_preserves_method_and_updates_only_scale() -> None:
    base_config = {
        "seed": 1234,
        "gw_data": {
            "cache_path": "data/processed/small_waveform_mismatch.h5",
            "num_waveforms": 256,
            "overwrite": True,
        },
        "pairs": {"num_pairs": 8192},
        "model": {"input_dim": 5, "embedding_dim": 4, "hidden_dims": [32, 16]},
        "training": {"epochs": 10, "batch_size": 128, "learning_rate": 0.001},
    }

    config = config_for_bank_size(
        base_config,
        2048,
        "data/processed/waveform_bank_{bank_size}_mismatch.h5",
        180.0,
    )

    assert config["gw_data"]["num_waveforms"] == 2048
    assert config["gw_data"]["cache_path"] == "data/processed/waveform_bank_2048_mismatch.h5"
    assert config["gw_data"]["overwrite"] is False
    assert config["pairs"]["num_pairs"] == 2048 * 32
    assert config["model"] == base_config["model"]
    assert config["training"] == base_config["training"]


def test_failed_record_keeps_bank_metadata() -> None:
    config = {
        "gw_data": {"cache_path": "data/processed/missing.h5"},
    }

    record = failed_record(config, 4096, RuntimeError("too expensive"))

    assert record["status"] == "failed"
    assert record["num_waveform_pairs"] == 4096 * 4095 // 2
    assert record["error_type"] == "RuntimeError"


def test_summary_files_are_readable(tmp_path: Path) -> None:
    records = [
        {
            "status": "completed",
            "bank_size": 128,
            "pearson": 0.9,
            "total_runtime_seconds": 1.5,
        },
        {
            "status": "failed",
            "bank_size": 8192,
            "error": "stopped",
        },
    ]

    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    write_summary(summary_path, records)
    write_summary_csv(csv_path, records)

    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    assert summary["num_completed"] == 1
    assert summary["num_failed"] == 1
    assert "pearson" in csv_path.read_text(encoding="utf-8")
