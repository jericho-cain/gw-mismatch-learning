# Classical MDS Baseline

## Status

Complete. The isolated results are in `outputs/mds_baseline/`. No manuscript or frozen
artifact was modified.

## Design

Classical MDS is a deterministic, non-parametric in-sample reference embedding constructed
directly from the complete mismatch matrix. It is evaluated at dimensions 2, 3, 4, 8, and
16 with the same Phase I geometry and retrieval metrics. It is not called an upper bound.

The centered Gram matrix is formed without explicitly materializing the centering matrix.
Leading coordinates use a partial symmetric eigensolver. A separate complete eigenvalue
calculation records positive, near-zero, and negative spectral counts and masses, because
the mismatch dissimilarity need not be Euclidean.

Classical MDS has no parametric map for unseen queries in this repository. No Phase III
or match-penalty comparison is therefore performed.

## Numerical findings

The N=8192 run completed in 94.80 s with 4321 MB peak memory. The centered Gram matrix has
4082 positive, 4108 negative, and 2 near-zero eigenvalues at the configured relative
tolerance. Positive spectral mass is 1465.36 and absolute negative mass is 148.42, giving
a negative-to-positive mass ratio of 0.1013. The most negative eigenvalue is -11.8454.
This substantial indefinite component confirms that mismatch is non-Euclidean.

At k=4, MDS gives Pearson 0.9409, RMSE 0.1076, Recall@10 0.6724, and Best@10 0.9888.
The encoder five-seed means are 0.9705, 0.0561, 0.7233, and 0.9765. Thus the encoder is
better for correlation, error, and Recall@10, while MDS is slightly better for Best@10.

At k=8, MDS gives Pearson 0.9849, RMSE 0.0423, Recall@10 0.7680, and Best@10 0.9984.
The encoder means are 0.9922, 0.0249, 0.7268, and 0.9813. The encoder remains better for
global fidelity, while MDS is better for retrieval. At k=16, MDS exceeds the encoder in
all four primary metrics, including Recall@10 0.8545 versus 0.7401.

MDS therefore does not act as a universal low-dimensional upper bound. It strengthens the
case that the learned representation is unusually effective for global fidelity at k=4
and k=8, while showing additional in-sample retrieval headroom at k=8 and especially k=16.
The most useful paper figure is the four-metric MDS-versus-encoder comparison, with a compact
table for k=4, 8, and 16. Future manuscript text should add this non-parametric in-sample
reference, explain the indefinite spectrum, and avoid any Phase III comparison.

## Validation

On frozen N=128, 256, and 512 banks, partial and full leading eigenvalues agree to at most
7.11e-15 absolute error and reconstructed distances agree exactly at stored float32
precision. Ruff and all 54 tests pass. Before/after frozen manifests are identical.

## Commands

```bash
.venv-gw/bin/python scripts/run_mds_baseline.py --validate-only
.venv-gw/bin/python scripts/run_mds_baseline.py
```
