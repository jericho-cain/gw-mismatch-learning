import matplotlib.pyplot as plt
import numpy as np

from gw_mismatch_learning.evaluation.plots import (
    plot_distance_scatter,
    plot_retrieval_rank_heatmap,
)
from gw_mismatch_learning.evaluation.retrieval import true_neighbors_from_distance


def test_plot_helpers_return_figures() -> None:
    distance = np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]])
    fig, _ = plot_distance_scatter(distance, distance)
    assert fig is not None
    plt.close(fig)

    neighbors = true_neighbors_from_distance(distance)
    fig, _ = plot_retrieval_rank_heatmap(neighbors, neighbors, top_k=2)
    assert fig is not None
    plt.close(fig)
