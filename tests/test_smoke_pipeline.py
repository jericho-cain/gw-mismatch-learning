import json
from pathlib import Path

import yaml
from scripts.train_embedding import run


def test_complete_synthetic_smoke_pipeline_generates_metrics_and_plots(tmp_path: Path) -> None:
    config = {
        "seed": 11,
        "synthetic_data": {
            "num_samples": 24,
            "input_dim": 16,
            "metric": "cosine",
            "noise": 0.0,
        },
        "pairs": {"num_pairs": 96},
        "model": {"input_dim": 16, "embedding_dim": 4, "hidden_dims": [16]},
        "training": {"epochs": 2, "batch_size": 24, "learning_rate": 0.001},
        "evaluation": {"top_k": 3},
        "outputs": {"plot_dir": str(tmp_path), "save_plots": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run(config_path)

    assert "pearson" in report
    assert "spearman" in report
    assert "recall_at_3" in report
    assert report["candidate_reduction_factor"] == 8.0
    assert (tmp_path / "distance_scatter.png").exists()
    assert (tmp_path / "retrieval_heatmap.png").exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config_path"] == str(config_path)
    assert "recall_at_3" in metrics["metrics"]
