# Phase I: Scaling Validation

Date: 2026-06-30

This note records the Phase I paper-validation experiment: a scaling study of
the current learned mismatch representation as waveform-bank size increases.
The goal was scientific validation, not model development. The learning
formulation, distance-regression objective, model architecture, optimizer,
latent dimension, waveform generation, PSD, mismatch computation, sampling
strategy, and evaluation metrics were kept fixed.

## Question

Does the learned latent representation continue to preserve pairwise
matched-filter mismatch relationships as the finite sampled waveform bank grows?

## Organization

This paper-validation sequence is separate from the earlier pipeline milestone
numbering. The old Phase 2 and Phase 3 notes now live under:

```text
docs/pipeline_milestones/
```

The new paper-validation sequence is:

- Phase I: Scaling validation, this note
- Phase II: Stronger physical-coordinate baselines
- Phase III: Candidate retrieval

## Code and Configuration

Runner:

```text
scripts/run_scaling_validation.py
```

Scaling config:

```text
configs/scaling_validation.yaml
```

Base config:

```text
configs/waveform_bank_small.yaml
```

The scaling config used:

```yaml
base_config: configs/waveform_bank_small.yaml
output_dir: outputs/scaling_validation
bank_sizes: [128, 256, 512, 1024, 2048, 4096, 8192]
cache_template: "data/processed/waveform_bank_{bank_size}_mismatch.h5"
max_estimated_runtime_minutes: 180.0
```

Tests covering the scaling runner are in:

```text
tests/test_scaling_validation.py
```

## Fixed Experimental Protocol

The independent variable was waveform-bank size only.

Fixed waveform and mismatch settings:

- Approximant: `IMRPhenomD`
- Binary type: nonspinning compact binaries
- PSD: `aLIGOZeroDetHighPower`
- Match routine: `pycbc.filter.match`
- Mismatch definition: `mismatch = 1 - match`
- Cache format: HDF5 mismatch matrices under `data/processed/`

Fixed model and training settings are inherited from
`configs/waveform_bank_small.yaml`:

- Input features: `(m1, m2, M, eta, chirp_mass)`
- Encoder input dimension: 5
- Latent dimension: 4
- Hidden dimensions: `[32, 16]`
- Optimizer: Adam
- Learning rate: 0.001
- Loss: distance regression against matched-filter mismatch

The only physical-coordinate baseline used in Phase I was the existing
standardized `(m1, m2)` baseline. No stronger baselines were introduced in this
phase.

## Outputs

Required tables:

```text
outputs/scaling_validation/runs.csv
outputs/scaling_validation/summary.csv
outputs/scaling_validation/summary.json
```

Per-bank metrics:

```text
outputs/scaling_validation/bank_size_128/metrics.json
outputs/scaling_validation/bank_size_256/metrics.json
outputs/scaling_validation/bank_size_512/metrics.json
outputs/scaling_validation/bank_size_1024/metrics.json
outputs/scaling_validation/bank_size_2048/metrics.json
outputs/scaling_validation/bank_size_4096/metrics.json
outputs/scaling_validation/bank_size_8192/metrics.json
```

Figures:

```text
outputs/scaling_validation/figures/pearson_vs_bank_size.png
outputs/scaling_validation/figures/spearman_vs_bank_size.png
outputs/scaling_validation/figures/mae_vs_bank_size.png
outputs/scaling_validation/figures/rmse_vs_bank_size.png
outputs/scaling_validation/figures/recall_at_k_vs_bank_size.png
outputs/scaling_validation/figures/best_match_recovery_vs_bank_size.png
outputs/scaling_validation/figures/runtime_vs_bank_size.png
outputs/scaling_validation/figures/pairwise_mismatch_time_vs_bank_size.png
outputs/scaling_validation/figures/physical_baseline_recall_at_k_vs_bank_size.png
outputs/scaling_validation/figures/physical_baseline_best_match_recovery_vs_bank_size.png
outputs/scaling_validation/figures/learned_vs_physical_recall.png
outputs/scaling_validation/figures/learned_vs_physical_best_match_recovery.png
```

Generated caches:

```text
data/processed/waveform_bank_128_mismatch.h5
data/processed/waveform_bank_256_mismatch.h5
data/processed/waveform_bank_512_mismatch.h5
data/processed/waveform_bank_1024_mismatch.h5
data/processed/waveform_bank_2048_mismatch.h5
data/processed/waveform_bank_4096_mismatch.h5
data/processed/waveform_bank_8192_mismatch.h5
```

The 128 through 1024 caches were reused. The 2048, 4096, and 8192 caches were
generated during this study.

## Results

All seven target bank sizes completed.

| Bank size | Pairs | Pearson | Spearman | MAE | RMSE | Recall@10 | Best@10 | Physical Recall@10 | Physical Best@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8,128 | 0.8971 | 0.9030 | 0.0737 | 0.1007 | 0.6211 | 0.8359 | 0.4008 | 0.5391 |
| 256 | 32,640 | 0.8986 | 0.8938 | 0.0766 | 0.1020 | 0.4992 | 0.8203 | 0.2965 | 0.6055 |
| 512 | 130,816 | 0.9362 | 0.9284 | 0.0618 | 0.0804 | 0.4873 | 0.8242 | 0.2510 | 0.5586 |
| 1024 | 523,776 | 0.9668 | 0.9620 | 0.0428 | 0.0589 | 0.6493 | 0.9561 | 0.2397 | 0.6094 |
| 2048 | 2,096,128 | 0.9694 | 0.9654 | 0.0418 | 0.0570 | 0.6301 | 0.9561 | 0.2335 | 0.6162 |
| 4096 | 8,386,560 | 0.9697 | 0.9567 | 0.0411 | 0.0550 | 0.7090 | 0.9775 | 0.2382 | 0.5935 |
| 8192 | 33,550,336 | 0.9704 | 0.9594 | 0.0418 | 0.0560 | 0.7552 | 0.9873 | 0.2410 | 0.5859 |

Neighborhood preservation at all evaluated K values:

| Bank size | Recall@5 | Recall@10 | Recall@20 | Best@5 | Best@10 | Best@20 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.4766 | 0.6211 | 0.8066 | 0.6875 | 0.8359 | 0.9453 |
| 256 | 0.4375 | 0.4992 | 0.6541 | 0.6758 | 0.8203 | 0.9180 |
| 512 | 0.4578 | 0.4873 | 0.5620 | 0.6816 | 0.8242 | 0.9160 |
| 1024 | 0.6076 | 0.6493 | 0.7013 | 0.8945 | 0.9561 | 0.9844 |
| 2048 | 0.6096 | 0.6301 | 0.6590 | 0.8945 | 0.9561 | 0.9839 |
| 4096 | 0.7017 | 0.7090 | 0.7092 | 0.9331 | 0.9775 | 0.9917 |
| 8192 | 0.7436 | 0.7552 | 0.7448 | 0.9568 | 0.9873 | 0.9976 |

Runtime accounting:

| Bank size | Pairwise mismatch time | Training time | Evaluation time | Total runtime | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0 s | 0.8 s | 0.0 s | 0.9 s | 780.8 MB |
| 256 | 0.0 s | 0.4 s | 0.0 s | 0.4 s | 784.7 MB |
| 512 | 0.0 s | 0.8 s | 0.1 s | 0.9 s | 795.8 MB |
| 1024 | 0.0 s | 1.6 s | 0.3 s | 1.9 s | 841.6 MB |
| 2048 | 548.1 s | 3.4 s | 1.2 s | 555.2 s | 1932.5 MB |
| 4096 | 2164.0 s | 6.8 s | 5.5 s | 2177.9 s | 3594.1 MB |
| 8192 | 8567.8 s | 13.1 s | 24.1 s | 8608.1 s | 8605.5 MB |

For cached banks, pairwise mismatch time is reported as `0.0 s` because no new
mismatch matrix was computed during this Phase I run.

## Interpretation

The learned latent representation continues to preserve pairwise matched-filter
mismatch relationships through the largest completed bank, 8192 waveforms.

Global geometry improves from the smallest banks and stabilizes for larger
banks. Pearson correlation reaches approximately `0.97` for 2048 through 8192,
and MAE remains near `0.041` over the same range.

Neighborhood preservation does not degrade with scale in this study. Learned
Recall@10 increases from `0.6301` at 2048 to `0.7090` at 4096 and `0.7552` at
8192. Best-match recovery@10 increases from `0.9561` at 2048 to `0.9775` at
4096 and `0.9873` at 8192.

The standardized `(m1, m2)` baseline remains substantially weaker at larger
bank sizes. Physical Recall@10 remains near `0.23` to `0.24` from 2048 through
8192, while learned Recall@10 is `0.6301` to `0.7552`.

Runtime scaling is dominated by pairwise matched-filter mismatch computation,
as expected. The pairwise mismatch stage grows from `548.1 s` at 2048 to
`2164.0 s` at 4096 and `8567.8 s` at 8192. Training and evaluation remain small
relative to mismatch generation.

## Validation

The completed run was followed by:

```bash
.venv-gw/bin/python -m ruff check .
.venv-gw/bin/python -m pytest
```

Validation status:

- Ruff: all checks passed
- Pytest: 34 passed
- `runs.csv`, `summary.csv`, and `summary.json` were readable
- All 12 PNG figures were readable

The only warning observed during artifact inspection was Matplotlib creating a
temporary cache directory because `/home/gravlab/.config/matplotlib` was not
writable. This did not affect generated artifacts.

## Conclusion

Phase I supports the claim that, within the tested nonspinning `IMRPhenomD`
bank and fixed implementation protocol, the learned representation continues to
preserve pairwise matched-filter mismatch relationships as bank size increases
through 8192 waveforms.

The next scientific question is not whether the method beats raw `(m1, m2)`;
Phase I confirms that it does at these scales. The next question is whether the
learned representation remains competitive against stronger physically
motivated coordinate baselines.
