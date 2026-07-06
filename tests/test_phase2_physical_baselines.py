import numpy as np
from scripts.run_phase2_physical_baselines import (
    raw_coordinate_features,
    standardized_coordinate_features,
    zscore,
)

from gw_mismatch_learning.datasets.gw import DistanceMatrixDataset


def toy_dataset() -> DistanceMatrixDataset:
    mass_1 = np.asarray([30.0, 36.0, 40.0], dtype=np.float32)
    mass_2 = np.asarray([20.0, 18.0, 10.0], dtype=np.float32)
    return DistanceMatrixDataset(
        features=np.zeros((3, 5), dtype=np.float32),
        distance=np.zeros((3, 3), dtype=np.float32),
        metadata={"mass_1": mass_1, "mass_2": mass_2},
    )


def test_raw_coordinate_features_uses_q_as_m2_over_m1() -> None:
    features = raw_coordinate_features(toy_dataset(), ["M", "q"])

    np.testing.assert_allclose(features[:, 0], np.asarray([50.0, 54.0, 50.0]))
    np.testing.assert_allclose(features[:, 1], np.asarray([20.0 / 30.0, 18.0 / 36.0, 10.0 / 40.0]))


def test_standardized_coordinate_features_zscores_per_column() -> None:
    features = standardized_coordinate_features(toy_dataset(), ["m1", "m2"])

    np.testing.assert_allclose(features.mean(axis=0), np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(features.std(axis=0), np.ones(2), atol=1e-6)


def test_zscore_handles_constant_columns() -> None:
    features = zscore(np.ones((4, 2), dtype=np.float32))

    np.testing.assert_allclose(features, np.zeros((4, 2), dtype=np.float32))
