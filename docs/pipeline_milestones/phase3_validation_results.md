# Phase 3 Validation Results

This note freezes the first Phase 3 validation snapshot. The goal of this phase was to measure whether the Phase 2 result is stable enough to motivate paper-quality follow-up experiments, not to tune the model.

## Reproducibility

```bash
python scripts/run_validation_sweep.py --config configs/phase3_validation.yaml
python -m pytest
python -m ruff check .
```

The recorded run completed successfully with:

- `python scripts/run_validation_sweep.py --config configs/phase3_validation.yaml`: 634.37 seconds
- `python -m pytest`: 22 passed
- `python -m ruff check .`: all checks passed

## Configuration

The sweep used:

```text
configs/phase3_validation.yaml
```

with base training settings from:

```text
configs/waveform_bank_small.yaml
```

Base model and training setup:

- Encoder: MLP
- Input dimension: 5
- Baseline latent dimension: 4
- Baseline hidden dimensions: `[32, 16]`
- Baseline epochs: 10
- Batch size: 128
- Learning rate: 0.001
- Distance-regression loss

The sweep varied:

- Seeds: `101`, `202`, `303`, `404`, `505`
- Latent dimensions: `2`, `4`, `8`, `16`, `32`, `64`
- Architectures: `[16]`, `[32, 16]`, `[64, 32]`, `[64, 32, 16]`
- Epochs: `3`, `5`, `10`, `20`
- Bank sizes: `128`, `256`, `512`, `1024`

Evaluation used K values:

```text
K = 5, 10, 20
```

## Waveform Setup

- Approximant: `IMRPhenomD`
- Binary type: nonspinning compact binaries
- Mass sampling: random uniform component masses with `m2 <= m1`
- `m1` range: 20 to 40 solar masses
- `m2` range: 10 to 30 solar masses
- Frequency spacing: `delta_f = 0.0625`
- Lower frequency cutoff: `f_lower = 20.0 Hz`
- Final frequency: `f_final = 512.0 Hz`
- PSD: `aLIGOZeroDetHighPower`
- Match routine: `pycbc.filter.match`
- Mismatch definition: `mismatch = 1 - match`

The physical-parameter baseline used Euclidean nearest-neighbor retrieval in raw `(m1, m2)` space.

## Outputs

The Phase 3 sweep wrote:

```text
outputs/phase3_validation/runs.csv
outputs/phase3_validation/summary.csv
outputs/phase3_validation/summary.json
outputs/phase3_validation/figures/correlation_vs_latent_dim.png
outputs/phase3_validation/figures/recall_vs_latent_dim.png
outputs/phase3_validation/figures/performance_vs_bank_size.png
outputs/phase3_validation/figures/seed_variability.png
outputs/phase3_validation/figures/learned_vs_physical_baseline.png
```

Waveform mismatch caches were generated for:

```text
data/processed/waveform_bank_128_mismatch.h5
data/processed/waveform_bank_256_mismatch.h5
data/processed/waveform_bank_512_mismatch.h5
data/processed/waveform_bank_1024_mismatch.h5
```

## Seed Study

Mean +/- standard deviation over seeds `101`, `202`, `303`, `404`, and `505`:

| Metric | Learned | Physical `(m1, m2)` baseline |
| --- | ---: | ---: |
| Recall@5 | 0.4692 +/- 0.0276 | 0.2547 +/- 0.0000 |
| Recall@10 | 0.5338 +/- 0.0253 | 0.2965 +/- 0.0000 |
| Recall@20 | 0.6736 +/- 0.0250 | 0.3924 +/- 0.0000 |
| Best recovered match@5 | 0.7086 +/- 0.0341 | 0.4492 +/- 0.0000 |
| Best recovered match@10 | 0.8398 +/- 0.0217 | 0.6055 +/- 0.0000 |
| Best recovered match@20 | 0.9258 +/- 0.0141 | 0.7422 +/- 0.0000 |

Distance-regression metrics over the same seed study:

| Metric | Mean +/- std |
| --- | ---: |
| Pearson correlation | 0.9035 +/- 0.0121 |
| Spearman correlation | 0.8999 +/- 0.0096 |
| MAE | 0.0736 +/- 0.0050 |
| RMSE | 0.0998 +/- 0.0063 |

These results demonstrate that learned latent coordinates preserve matched-filter mismatch neighborhoods more effectively than naive Euclidean distance in ((m_1,m_2)) space for the tested nonspinning IMRPhenomD banks.

## Bank-Size Scaling

| Bank size | Pairwise overlaps | Runtime | Learned Recall@5 | Physical Recall@5 | Learned best@5 | Physical best@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8,128 | 9.3 s | 0.4766 | 0.2828 | 0.6875 | 0.4219 |
| 256 | 32,640 | 28.8 s | 0.4375 | 0.2547 | 0.6758 | 0.4492 |
| 512 | 130,816 | 114.1 s | 0.4578 | 0.2375 | 0.6816 | 0.4316 |
| 1024 | 523,776 | 456.8 s | 0.6049 | 0.2301 | 0.8877 | 0.4580 |

Runtime scaling is dominated by pairwise matched-filter overlap computation, as expected. The 1024-waveform mismatch matrix required 523,776 pairwise overlap computations and completed in approximately 7.5 minutes during this run.

## Latent-Dimension Study

| Latent dimension | Pearson | Spearman | MAE | RMSE | Learned Recall@5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.8922 | 0.8775 | 0.1032 | 0.1310 | 0.5977 |
| 4 | 0.8986 | 0.8938 | 0.0766 | 0.1020 | 0.4375 |
| 8 | 0.9202 | 0.9237 | 0.0623 | 0.0867 | 0.4094 |
| 16 | 0.9307 | 0.9234 | 0.0578 | 0.0797 | 0.4344 |
| 32 | 0.9399 | 0.9362 | 0.0534 | 0.0741 | 0.4453 |
| 64 | 0.9562 | 0.9603 | 0.0395 | 0.0610 | 0.5234 |

Correlation and distance-regression error improved through 64 latent dimensions in this single-seed sweep. Retrieval metrics were noisier and should not yet be interpreted as evidence of a stable optimum or saturation point.

## Architecture and Epoch Diagnostics

The architecture and epoch sweeps are diagnostic only. Wider or deeper MLPs improved retrieval relative to the baseline, and longer training improved correlation and distance error. These observations should not be treated as tuned model results.

The guiding rule for the next stage is still:

```text
measure first, optimize later
```

## Caveats

- The banks are nonspinning and use only `IMRPhenomD`.
- The mass range is deliberately narrow.
- The physical baseline is intentionally simple and uses only raw `(m1, m2)` Euclidean distance.
- The train/test setup is still a compact development-scale validation, not a production search setting.
- No runtime speedup claim is made.
- Exact matched filtering remains the ground-truth verification step.
- The latent-dimension, architecture, and epoch sweeps are preliminary and mostly single-seed diagnostics.

## Next Comparison

Before tuning the learned model, the next scientifically necessary comparison is against better physical coordinate systems while keeping the learned embedding unchanged.

Planned baselines:

- Chirp mass
- Symmetric mass ratio
- Total mass plus mass ratio

The next question is whether the learned latent representation still outperforms physically motivated parameterizations, not merely raw component-mass coordinates.
