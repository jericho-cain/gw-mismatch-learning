from __future__ import annotations

import os
from pathlib import Path

_MPLCONFIGDIR = Path("/tmp/gw_mismatch_learning_mpl")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gw_mismatch_learning.evaluation.geometry import upper_triangle_values  # noqa: E402


def plot_distance_scatter(latent_distance: np.ndarray, target_distance: np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(
        upper_triangle_values(target_distance),
        upper_triangle_values(latent_distance),
        s=8,
        alpha=0.6,
    )
    ax.set_xlabel("Target distance")
    ax.set_ylabel("Latent distance")
    return fig, ax


def plot_retrieval_rank_heatmap(
    true_neighbors: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
):
    overlap = np.zeros((len(true_neighbors), top_k), dtype=np.float32)
    for row, (true_row, retrieved_row) in enumerate(
        zip(true_neighbors, retrieved_neighbors, strict=True)
    ):
        true_set = set(true_row[:top_k])
        overlap[row] = [float(neighbor in true_set) for neighbor in retrieved_row[:top_k]]

    fig, ax = plt.subplots()
    image = ax.imshow(overlap, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Retrieved rank")
    ax.set_ylabel("Query index")
    ax.set_title(f"Top-{top_k} neighbor preservation")
    fig.colorbar(image, ax=ax, label="In target top-K")
    return fig, ax
