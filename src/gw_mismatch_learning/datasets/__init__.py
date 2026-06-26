"""Datasets and sampling utilities."""

from gw_mismatch_learning.datasets.distance_regression import DistanceRegressionDataset
from gw_mismatch_learning.datasets.pairs import PairBatch, PairDataset, sample_pairs
from gw_mismatch_learning.datasets.synthetic import (
    SyntheticDistanceDataset,
    make_synthetic_distance_dataset,
    pairwise_feature_distance,
    validate_distance_matrix,
)
from gw_mismatch_learning.datasets.triplets import TripletBatch, TripletDataset, sample_triplets

__all__ = [
    "DistanceRegressionDataset",
    "PairBatch",
    "PairDataset",
    "SyntheticDistanceDataset",
    "TripletBatch",
    "TripletDataset",
    "make_synthetic_distance_dataset",
    "pairwise_feature_distance",
    "sample_pairs",
    "sample_triplets",
    "validate_distance_matrix",
]
