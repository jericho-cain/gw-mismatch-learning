from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_distance_scatter(latent_distance: np.ndarray, target_distance: np.ndarray):
    fig, ax = plt.subplots()
    mask = np.triu(np.ones_like(target_distance, dtype=bool), k=1)
    ax.scatter(target_distance[mask], latent_distance[mask], s=8, alpha=0.6)
    ax.set_xlabel("Target mismatch")
    ax.set_ylabel("Latent distance")
    return fig, ax
