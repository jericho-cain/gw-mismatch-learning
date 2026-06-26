from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_embedding import run as run_training

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_validation.yaml")
    parser.add_argument(
        "--studies",
        nargs="*",
        help="Optional study names to run: seed, latent_dim, architecture, epochs, bank_size.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_config = load_config(args.config)
    selected = set(args.studies or [])
    runs = build_runs(sweep_config, selected_studies=selected)
    if args.dry_run:
        print(f"Planned runs: {len(runs)}")
        for run_spec in runs:
            print(run_spec["run_id"])
        return

    output_dir = ensure_dir(sweep_config.get("output_dir", "outputs/phase3_validation"))
    records = execute_runs(runs, output_dir)
    write_records(output_dir / "runs.csv", records)
    write_summary(output_dir / "summary.json", records)
    write_summary_csv(output_dir / "summary.csv", records)
    make_plots(output_dir, records, sweep_config)


def build_runs(sweep_config: dict[str, Any], selected_studies: set[str]) -> list[dict[str, Any]]:
    base_config_path = Path(sweep_config["base_config"])
    base_config = load_config(base_config_path)
    studies = sweep_config.get("studies", {})
    runs: list[dict[str, Any]] = []

    def enabled(name: str) -> bool:
        study = studies.get(name, {})
        return bool(study.get("enabled", False)) and (
            not selected_studies or name in selected_studies
        )

    if enabled("seed"):
        for seed in studies["seed"].get("seeds", []):
            config = copy.deepcopy(base_config)
            config["seed"] = int(seed)
            append_run(runs, "seed", f"seed_{seed}", config, {"seed": int(seed)})

    if enabled("latent_dim"):
        study = studies["latent_dim"]
        for value in study.get("values", []):
            config = copy.deepcopy(base_config)
            config["seed"] = int(study.get("seed", base_config.get("seed", 1234)))
            config["model"]["embedding_dim"] = int(value)
            append_run(
                runs,
                "latent_dim",
                f"latent_dim_{value}",
                config,
                {"latent_dim": int(value)},
            )

    if enabled("architecture"):
        study = studies["architecture"]
        for variant in study.get("variants", []):
            config = copy.deepcopy(base_config)
            config["seed"] = int(study.get("seed", base_config.get("seed", 1234)))
            config["model"]["hidden_dims"] = [int(value) for value in variant["hidden_dims"]]
            append_run(
                runs,
                "architecture",
                f"architecture_{variant['name']}",
                config,
                {
                    "architecture": variant["name"],
                    "hidden_dims": "-".join(str(value) for value in variant["hidden_dims"]),
                },
            )

    if enabled("epochs"):
        study = studies["epochs"]
        for value in study.get("values", []):
            config = copy.deepcopy(base_config)
            config["seed"] = int(study.get("seed", base_config.get("seed", 1234)))
            config["training"]["epochs"] = int(value)
            append_run(runs, "epochs", f"epochs_{value}", config, {"epochs": int(value)})

    if enabled("bank_size"):
        study = studies["bank_size"]
        for value in study.get("values", []):
            config = copy.deepcopy(base_config)
            config["seed"] = int(study.get("seed", base_config.get("seed", 1234)))
            config["gw_data"]["num_waveforms"] = int(value)
            config["gw_data"]["cache_path"] = f"data/processed/waveform_bank_{value}_mismatch.h5"
            config["pairs"]["num_pairs"] = max(int(config["pairs"]["num_pairs"]), int(value) * 32)
            append_run(runs, "bank_size", f"bank_size_{value}", config, {"bank_size": int(value)})

    return runs


def append_run(
    runs: list[dict[str, Any]],
    study: str,
    run_id: str,
    config: dict[str, Any],
    parameters: dict[str, Any],
) -> None:
    runs.append({"study": study, "run_id": run_id, "config": config, "parameters": parameters})


def execute_runs(runs: Iterable[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_spec in runs:
        run_dir = ensure_dir(output_dir / run_spec["study"] / run_spec["run_id"])
        config = copy.deepcopy(run_spec["config"])
        config["outputs"] = {
            **config.get("outputs", {}),
            "plot_dir": str(run_dir),
            "save_plots": False,
            "save_metrics": True,
        }
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"Running {run_spec['run_id']}...")
        start = time.perf_counter()
        metrics = run_training(config_path)
        runtime_seconds = time.perf_counter() - start
        metrics["runtime_seconds"] = runtime_seconds

        record = {
            "study": run_spec["study"],
            "run_id": run_spec["run_id"],
            **run_spec["parameters"],
            **metrics,
        }
        records.append(record)

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "study": run_spec["study"],
                    "run_id": run_spec["run_id"],
                    "parameters": run_spec["parameters"],
                    "runtime_seconds": runtime_seconds,
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Completed {run_spec['run_id']} in {runtime_seconds:.1f}s")
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    summary = summarize_records(records)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    summary = summarize_records(records)
    rows = []
    for study, metrics in summary.items():
        for metric, stats in metrics.items():
            rows.append({"study": study, "metric": metric, **stats})
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    parameter_keys = {"seed", "latent_dim", "epochs", "bank_size"}
    numeric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, int | float) and key not in parameter_keys
        }
    )
    studies = sorted({record["study"] for record in records})
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for study in studies:
        study_records = [record for record in records if record["study"] == study]
        summary[study] = {}
        for key in numeric_keys:
            values = [float(record[key]) for record in study_records if key in record]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
                summary[study][key] = {
                    "count": float(len(values)),
                    "mean": mean,
                    "std": variance**0.5,
                }
    return summary


def make_plots(
    output_dir: Path,
    records: list[dict[str, Any]],
    sweep_config: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "figures")
    metrics = sweep_config.get("plot_metrics", {})
    plot_metric_by_parameter(
        records,
        study="latent_dim",
        parameter="latent_dim",
        metric=metrics.get("correlation", "spearman"),
        output_path=plot_dir / "correlation_vs_latent_dim.png",
        ylabel="Correlation",
    )
    plot_metric_by_parameter(
        records,
        study="latent_dim",
        parameter="latent_dim",
        metric=metrics.get("recall", "learned_recall_at_5"),
        output_path=plot_dir / "recall_vs_latent_dim.png",
        ylabel="Recall@K",
    )
    plot_metric_by_parameter(
        records,
        study="bank_size",
        parameter="bank_size",
        metric=metrics.get("bank_size", "learned_recall_at_5"),
        output_path=plot_dir / "performance_vs_bank_size.png",
        ylabel="Performance",
    )
    plot_seed_variability(records, plot_dir / "seed_variability.png")
    plot_learned_vs_physical(
        records,
        learned_metric=metrics.get("recall", "learned_recall_at_5"),
        physical_metric=metrics.get("baseline_recall", "physical_parameter_recall_at_5"),
        output_path=plot_dir / "learned_vs_physical_baseline.png",
    )
    plt.close("all")


def plot_metric_by_parameter(
    records: list[dict[str, Any]],
    study: str,
    parameter: str,
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        [record for record in records if record.get("study") == study and metric in record],
        key=lambda record: record[parameter],
    )
    if not rows:
        return
    fig, ax = plt.subplots()
    ax.plot([record[parameter] for record in rows], [record[metric] for record in rows], marker="o")
    ax.set_xlabel(parameter.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seed_variability(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [record for record in records if record.get("study") == "seed"]
    metrics = ["spearman", "learned_recall_at_5", "physical_parameter_recall_at_5"]
    available = [metric for metric in metrics if any(metric in record for record in rows)]
    if not rows or not available:
        return
    means = []
    stds = []
    for metric in available:
        values = [float(record[metric]) for record in rows if metric in record]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        means.append(mean)
        stds.append(variance**0.5)
    fig, ax = plt.subplots()
    ax.bar(available, means, yerr=stds, capsize=4)
    ax.set_ylabel("Mean +/- std")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learned_vs_physical(
    records: list[dict[str, Any]],
    learned_metric: str,
    physical_metric: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = [record for record in records if learned_metric in record and physical_metric in record]
    if not rows:
        return
    labels = [record["run_id"] for record in rows]
    x_values = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.4), 4))
    ax.plot(x_values, [record[learned_metric] for record in rows], marker="o", label="Learned")
    ax.plot(
        x_values,
        [record[physical_metric] for record in rows],
        marker="o",
        label="Physical parameter",
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Recall@K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
