from gw_mismatch_learning.datasets.distance_regression import DistanceRegressionDataset
from gw_mismatch_learning.datasets.pairs import sample_pairs
from gw_mismatch_learning.datasets.synthetic import make_synthetic_distance_dataset
from gw_mismatch_learning.datasets.triplets import sample_triplets


def test_pair_and_distance_regression_dataset_shapes() -> None:
    synthetic = make_synthetic_distance_dataset(num_samples=10, input_dim=5, seed=1)
    pairs = sample_pairs(synthetic.features, synthetic.distance, num_pairs=12, seed=2)
    dataset = DistanceRegressionDataset(pairs)
    left, right, distance = dataset[0]

    assert len(dataset) == 12
    assert left.shape == right.shape == (5,)
    assert distance.ndim == 0
    assert pairs.left_index.shape == pairs.right_index.shape == (12,)


def test_triplet_sampler_uses_near_positive_and_far_negative() -> None:
    synthetic = make_synthetic_distance_dataset(num_samples=12, input_dim=4, seed=3)
    triplets = sample_triplets(synthetic.features, synthetic.distance, num_triplets=8, seed=4)
    assert triplets.anchor.shape == triplets.positive.shape == triplets.negative.shape
