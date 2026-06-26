import numpy as np

from gw_mismatch_learning.datasets.waveform_dataset import make_mock_waveform_dataset
from gw_mismatch_learning.evaluation.geometry import distance_correlations


def test_mock_mismatch_is_symmetric_with_zero_diagonal() -> None:
    dataset = make_mock_waveform_dataset(num_samples=12, input_dim=4, seed=1)
    assert dataset.features.shape == (12, 4)
    assert dataset.mismatch.shape == (12, 12)
    np.testing.assert_allclose(dataset.mismatch, dataset.mismatch.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(dataset.mismatch), 0.0)


def test_distance_correlations_smoke() -> None:
    distance = np.array([[0.0, 0.2, 0.5], [0.2, 0.0, 0.4], [0.5, 0.4, 0.0]], dtype=np.float32)
    report = distance_correlations(distance, distance)
    assert report["pearson"] > 0.99
    assert report["spearman"] > 0.99
