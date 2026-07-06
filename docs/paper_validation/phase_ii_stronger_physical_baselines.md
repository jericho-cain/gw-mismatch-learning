# Phase II: Stronger Physical Baselines

Date: 2026-07-06

This note records the Phase II paper-validation experiment: a fixed-coordinate
baseline study using the validated Phase I waveform banks and mismatch caches.
The goal was to test whether mismatch-supervised representation learning
improves neighborhood retrieval beyond the geometry already present in
physically motivated coordinate systems.

No mismatch matrices were regenerated. No learned model was retrained. Phase I
outputs were used only as frozen learned-reference metrics.

## Question

Does the learned representation preserve pairwise matched-filter mismatch
neighborhoods more faithfully than fixed physical coordinate systems
constructed from the same waveform parameters?

## Code and Configuration

Runner:

```text
scripts/run_phase2_physical_baselines.py
```

Configuration:

```text
configs/phase2_physical_baselines.yaml
```

Tests:

```text
tests/test_phase2_physical_baselines.py
```

Reference Phase I outputs:

```text
outputs/scaling_validation/
```

Phase II outputs:

```text
outputs/phase2_physical_baselines/
```

## Protocol

Primary bank sizes:

```text
1024, 2048, 4096, 8192
```

All coordinate systems used the same waveform samples, mismatch matrices,
query waveforms, K values, and self-match exclusion convention. Each fixed
coordinate system was standardized per bank using column-wise z-score
standardization.

K values:

```text
K = 5, 10, 20
```

The mass-ratio convention was:

```text
q = m2 / m1 with m1 >= m2, so 0 < q <= 1
```

Evaluated coordinate systems:

| Coordinate system | Features | Dimensionality |
| --- | --- | ---: |
| Learned latent | frozen Phase I latent coordinates | 4 |
| Component masses | `(m1, m2)` | 2 |
| Chirp mass + eta | `(chirp_mass, eta)` | 2 |
| Total mass + q | `(M, q)` | 2 |
| Total mass + eta | `(M, eta)` | 2 |
| Five-feature input | `(m1, m2, M, eta, chirp_mass)` | 5 |

The five-feature input space is a control, not a learned baseline. It tests
whether the supervised encoder improves retrieval beyond the standardized
engineered features it receives as input.

## Outputs

Tables:

```text
outputs/phase2_physical_baselines/runs.csv
outputs/phase2_physical_baselines/summary.csv
outputs/phase2_physical_baselines/summary.json
```

Figures:

```text
outputs/phase2_physical_baselines/figures/recall_at_k_by_coordinate_system_8192.png
outputs/phase2_physical_baselines/figures/best_match_recovery_by_coordinate_system_8192.png
outputs/phase2_physical_baselines/figures/recall_at_10_vs_bank_size_by_coordinate_system.png
outputs/phase2_physical_baselines/figures/best_match_recovery_at_10_vs_bank_size_by_coordinate_system.png
outputs/phase2_physical_baselines/figures/learned_latent_vs_five_feature_input.png
```

## Results at 8192

The 8192-waveform bank is the primary Phase II result.

| Coordinate system | Recall@5 | Recall@10 | Recall@20 | Best@5 | Best@10 | Best@20 | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned latent | 0.7436 | 0.7552 | 0.7448 | 0.9568 | 0.9873 | 0.9976 | 0.9704 | 0.9594 |
| Component masses | 0.2321 | 0.2410 | 0.2394 | 0.4432 | 0.5859 | 0.7317 | 0.5369 | 0.5725 |
| Chirp mass + eta | 0.2760 | 0.2853 | 0.2893 | 0.5111 | 0.6656 | 0.8149 | 0.5125 | 0.5767 |
| Total mass + q | 0.2366 | 0.2429 | 0.2395 | 0.4443 | 0.5793 | 0.7214 | 0.5298 | 0.5660 |
| Total mass + eta | 0.2896 | 0.3006 | 0.3029 | 0.5330 | 0.6919 | 0.8387 | 0.5566 | 0.6142 |
| Five-feature input | 0.2583 | 0.2685 | 0.2685 | 0.4878 | 0.6410 | 0.7888 | 0.6055 | 0.6495 |

The strongest fixed physical coordinate system at 8192 was `(M, eta)` for both
Recall@K and best-match recovery@K. It did not approach the learned latent
representation.

## Bank-Size Stability

Recall@10:

| Bank size | Learned latent | Component masses | Chirp mass + eta | Total mass + q | Total mass + eta | Five-feature input |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.6493 | 0.2397 | 0.2886 | 0.2289 | 0.3014 | 0.2692 |
| 2048 | 0.6301 | 0.2335 | 0.2907 | 0.2299 | 0.3060 | 0.2646 |
| 4096 | 0.7090 | 0.2382 | 0.2898 | 0.2418 | 0.3041 | 0.2680 |
| 8192 | 0.7552 | 0.2410 | 0.2853 | 0.2429 | 0.3006 | 0.2685 |

Best-match recovery@10:

| Bank size | Learned latent | Component masses | Chirp mass + eta | Total mass + q | Total mass + eta | Five-feature input |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.9561 | 0.6094 | 0.6562 | 0.5830 | 0.6836 | 0.6514 |
| 2048 | 0.9561 | 0.6162 | 0.7021 | 0.6035 | 0.7241 | 0.6626 |
| 4096 | 0.9775 | 0.5935 | 0.6794 | 0.5852 | 0.7029 | 0.6501 |
| 8192 | 0.9873 | 0.5859 | 0.6656 | 0.5793 | 0.6919 | 0.6410 |

The conclusion is stable over bank size. The learned latent representation
outperforms every fixed coordinate system at every tested bank size and every
tested K value.

## Five-Feature Control

The five-feature input space is consistently stronger than raw component
masses for correlation, but it is not the strongest retrieval baseline. At
8192:

- Five-feature Recall@10: `0.2685`
- Best fixed physical Recall@10, `(M, eta)`: `0.3006`
- Learned latent Recall@10: `0.7552`

For best-match recovery at 8192:

- Five-feature Best@10: `0.6410`
- Best fixed physical Best@10, `(M, eta)`: `0.6919`
- Learned latent Best@10: `0.9873`

This shows that the mismatch-supervised encoder provides a substantial
neighborhood-preservation improvement beyond the geometry already present in
the standardized engineered input features.

## Required Answers

1. Best physical coordinate system for Recall@K:
   `(M, eta)` is strongest across the tested banks and K values.

2. Best physical coordinate system for best-match recovery@K:
   `(M, eta)` is strongest across the tested banks and K values.

3. Five-feature input space versus learned latent space:
   The five-feature input space is far weaker than the learned latent space for
   both Recall@K and best-match recovery@K. At 8192, learned Recall@10 is
   `0.7552` versus `0.2685` for the five-feature input space.

4. Does the learned representation outperform every tested fixed physical
   coordinate system?
   Yes. It outperforms every tested fixed coordinate system for every tested
   bank size and K value.

5. Does the conclusion remain stable with bank size?
   Yes. The learned representation remains clearly ahead from 1024 through
   8192.

6. Are there any K values or bank sizes where a physical baseline matches or
   exceeds the learned representation?
   No. No fixed physical baseline matches or exceeds the learned representation
   in this Phase II study.

## Validation

Phase II validation performed:

```bash
.venv-gw/bin/python scripts/run_phase2_physical_baselines.py --config configs/phase2_physical_baselines.yaml
.venv-gw/bin/python -m ruff check .
.venv-gw/bin/python -m pytest
```

Artifact checks:

- `runs.csv`, `summary.csv`, and `summary.json` were readable
- All 5 generated PNG figures were readable
- No files under `outputs/scaling_validation/` had modification times on
  2026-07-06, confirming the Phase I outputs were not modified by this run

The only warning observed during figure verification was Matplotlib creating a
temporary cache directory because `/home/gravlab/.config/matplotlib` was not
writable. This did not affect generated artifacts.

## Conclusion

Phase II answers the baseline question directly: mismatch-supervised
representation learning provides substantially better matched-filter
neighborhood retrieval than fixed physically motivated coordinate systems
constructed from the same waveform parameters.

The strongest fixed baseline is `(M, eta)`, but it remains far below the
learned latent representation. The five-feature control confirms that the
learned transformation, not merely the engineered input feature set, is
responsible for the neighborhood-preservation improvement.
