import numpy as np

from gw_mismatch_learning.evaluation.retrieval import retrieval_report
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex


def test_sklearn_neighbor_retrieval_smoke() -> None:
    embeddings = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float32)
    mismatch = np.abs(embeddings - embeddings.T)
    index = SklearnNeighborIndex(embeddings)
    _, neighbors = index.query(embeddings, top_k=2)
    report = retrieval_report(mismatch, neighbors, top_k=2)
    assert neighbors.shape == (4, 2)
    assert report["candidate_reduction_factor"] == 2.0
