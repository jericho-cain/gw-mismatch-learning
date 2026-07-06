# Phase III: Out-of-Sample Candidate Retrieval

Date: 2026-07-06

This note records the Phase III paper-validation experiment: an out-of-sample
candidate-retrieval study using exact matched filtering as the reference. Phase
I and Phase II were frozen before this run at commit:

```text
b8e50c4aa26eb0ba515a38c741d656a557cdd493
```

No Phase I or Phase II outputs, documentation, checksums, or freeze artifacts
were modified during this phase.

## Question

Can the learned latent representation reduce the number of bank templates
requiring exact matched-filter comparison while preserving the result of an
exhaustive matched-filter search for query waveforms not used to construct or
train the representation?

## Code and Configuration

Runner:

```text
scripts/run_phase3_candidate_retrieval.py
```

Config:

```text
configs/phase3_candidate_retrieval.yaml
```

Tests:

```text
tests/test_phase3_candidate_retrieval.py
```

The run used the frozen 8192-waveform bank:

```text
data/processed/waveform_bank_8192_mismatch.h5
```

The 8192 mismatch matrix was reused and was not regenerated.

## Fixed Protocol

Bank and waveform settings:

- Bank size: 8192
- Query waveforms: 512
- Query seed: `9876`
- Approximant: `IMRPhenomD`
- Binary type: nonspinning compact binaries
- PSD: `aLIGOZeroDetHighPower`
- `delta_f`: `0.0625`
- `f_lower`: `20.0`
- `f_final`: `512.0`
- Mass ranges: same as the frozen 8192 bank
- Ordered masses: same convention as the bank, with `m1 >= m2`

Candidate K values:

```text
1, 2, 5, 10, 20, 50, 100, 200, 500
```

Methods:

- Learned latent retrieval
- Total mass + eta, the strongest fixed Phase II physical baseline
- Five-feature input-space baseline, using `(m1, m2, M, eta, chirp_mass)`

All methods used the same query waveforms, bank templates, K values, and
exhaustive reference matches.

## Query Handling

Query waveforms were newly sampled and were not members of the bank. Exact mass
duplicates of bank points were rejected during query sampling.

For each query, the runner computed the same engineered feature vector used by
the encoder:

```text
(m1, m2, M, eta, chirp_mass)
```

Query features were standardized using the bank feature means and standard
deviations. No scaler was fit on query features.

## Reference Search

For each query waveform, exact PyCBC matches were computed against all 8192 bank
templates. The exhaustive best template and match were used as the reference for
candidate retrieval.

Total exhaustive match evaluations:

```text
512 * 8192 = 4,194,304
```

## Outputs

Required tables:

```text
outputs/phase3_candidate_retrieval/runs.csv
outputs/phase3_candidate_retrieval/summary.csv
outputs/phase3_candidate_retrieval/summary.json
outputs/phase3_candidate_retrieval/query_results.csv
```

Figures:

```text
outputs/phase3_candidate_retrieval/figures/exact_best_recovery_vs_candidate_fraction.png
outputs/phase3_candidate_retrieval/figures/delta_match_vs_candidate_fraction.png
outputs/phase3_candidate_retrieval/figures/reduction_vs_exact_best_recovery.png
outputs/phase3_candidate_retrieval/figures/method_comparison_exact_best_recovery.png
outputs/phase3_candidate_retrieval/figures/candidate_vs_exhaustive_evaluations.png
```

## Results

Exact-best recovery rate:

| K | Candidate fraction | Match-eval reduction | Learned latent | Total mass + eta | Five-feature input |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000122 | 8192.0x | 0.7246 | 0.2383 | 0.2031 |
| 2 | 0.000244 | 4096.0x | 0.8672 | 0.3398 | 0.3184 |
| 5 | 0.000610 | 1638.4x | 0.9570 | 0.5234 | 0.4746 |
| 10 | 0.001221 | 819.2x | 0.9902 | 0.7168 | 0.6465 |
| 20 | 0.002441 | 409.6x | 1.0000 | 0.8496 | 0.7969 |
| 50 | 0.006104 | 163.8x | 1.0000 | 0.9609 | 0.9395 |
| 100 | 0.012207 | 81.9x | 1.0000 | 0.9980 | 0.9707 |
| 200 | 0.024414 | 41.0x | 1.0000 | 1.0000 | 0.9941 |
| 500 | 0.061035 | 16.4x | 1.0000 | 1.0000 | 1.0000 |

Learned latent match-loss statistics:

| K | Mean DeltaM | Median DeltaM | 95th pct DeltaM | 99th pct DeltaM | Max DeltaM | Fraction DeltaM <= 1e-3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.017e-04 | 0.0 | 5.464e-04 | 1.915e-03 | 4.030e-03 | 0.9727 |
| 2 | 3.697e-05 | 0.0 | 1.667e-04 | 7.235e-04 | 3.679e-03 | 0.9922 |
| 5 | 6.300e-06 | 0.0 | 0.0 | 2.281e-04 | 5.663e-04 | 1.0000 |
| 10 | 5.675e-07 | 0.0 | 0.0 | 0.0 | 2.153e-04 | 1.0000 |
| 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0000 |
| 50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0000 |
| 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0000 |
| 200 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0000 |
| 500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0000 |

At K=10, learned latent retrieval recovered the exhaustive best-matching
template for `99.02%` of out-of-sample queries while requiring exact
matched-filter evaluation of only `0.122%` of the bank. The largest observed
match loss at K=10 was `2.153e-04`.

At K=20, learned latent retrieval recovered the exhaustive best-matching
template for all 512 out-of-sample queries while requiring exact matched-filter
evaluation of `0.244%` of the bank, a `409.6x` matched-filter evaluation
reduction.

## Baseline Comparison

The learned latent representation outperformed both physical-coordinate
baselines at small and moderate K.

At K=10:

- Learned latent exact-best recovery: `0.9902`
- Total mass + eta exact-best recovery: `0.7168`
- Five-feature input exact-best recovery: `0.6465`

At K=20:

- Learned latent exact-best recovery: `1.0000`
- Total mass + eta exact-best recovery: `0.8496`
- Five-feature input exact-best recovery: `0.7969`

The strongest physical baseline, total mass + eta, required K=200 to reach
perfect exact-best recovery in this run. The learned latent method reached
perfect recovery at K=20.

## Runtime Accounting

Overall run timing:

| Stage | Time |
| --- | ---: |
| Query waveform generation | 1.81 s |
| Bank waveform generation | 3.30 s |
| Encoder training | 13.83 s |
| Exhaustive matching | 1083.01 s |
| Latent embedding | 0.0004 s |
| Total run | 1463.74 s |
| Peak memory | 4454.1 MB |

Per-method candidate matching across all K values:

| Method | Candidate matching time, all K | Retrieval time |
| --- | ---: | ---: |
| Learned latent | 119.33 s | 0.022 s |
| Total mass + eta | 118.42 s | 0.021 s |
| Five-feature input | 119.92 s | 0.023 s |

The dominant runtime cost is exact matched filtering. The measured wall-clock
times should be interpreted as validation accounting for this script and
machine, not as a general search-acceleration claim.

## Interpretation

Phase III supports the claim that the learned latent representation can act as a
candidate-selection front end for exact matched filtering on out-of-sample
queries from the same nonspinning `IMRPhenomD` parameter range.

The best practical operating point in this run is K=20: it gives perfect
exact-best recovery for the 512 evaluated queries while reducing exact
matched-filter evaluations by `409.6x` relative to exhaustive matching. K=10 is
also a strong operating point when a small miss rate is acceptable: it gives
`99.02%` exact-best recovery with an `819.2x` match-evaluation reduction and
max observed DeltaM of `2.153e-04`.

The result should be described as candidate reduction or matched-filter
evaluation reduction. It should not be described as detection acceleration
without additional end-to-end runtime studies.

## Limitations

This phase used out-of-sample queries drawn from the same distribution and
waveform settings as the training bank. It does not test domain shift, spinning
systems, different approximants, different PSDs, or larger banks. The learned
encoder was retrained from the frozen Phase I protocol rather than loaded from a
stored checkpoint, because Phase I did not freeze a model checkpoint artifact.

## Validation

The completed run was followed by:

```bash
.venv-gw/bin/python -m ruff check .
.venv-gw/bin/python -m pytest
```

Validation status:

- Ruff: all checks passed
- Pytest: 42 passed
- `runs.csv`, `summary.csv`, `summary.json`, and `query_results.csv` were readable
- All 5 PNG figures were readable
- Phase I and Phase II checksums still verify
- No Phase I or Phase II output files were modified after Phase III began
- Files modified under `outputs/` after Phase III began were confined to
  `outputs/phase3_candidate_retrieval/`

The only warning observed during figure inspection was Matplotlib creating a
temporary cache directory because `/home/gravlab/.config/matplotlib` was not
writable. This did not affect generated artifacts.
