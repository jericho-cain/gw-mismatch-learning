# Phase 3 Milestone

Date: 2026-06-26

This entry freezes the repository at the first major scientific validation milestone. It is a lab notebook record, not a paper draft.

## Project Objective

This project asks whether the geometry induced by matched-filter mismatch can be learned as a latent representation of waveform space.

The goal is not to replace matched filtering. Exact matched filtering remains the ground truth. The learned representation is evaluated as a way to organize waveform space so that Euclidean distance in latent coordinates approximates matched-filter mismatch neighborhoods.

## Current Architecture

The repository now has a modular distance-matrix learning pipeline:

- `waveforms/`: waveform-bank construction and PyCBC/LALSuite waveform generation helpers
- `metrics/`: synthetic metric and overlap-related utilities
- `datasets/`: synthetic, pair, triplet, distance-regression, and GW mismatch adapters
- `models/`: MLP encoder and distance-regression training
- `retrieval/`: nearest-neighbor retrieval with scikit-learn
- `evaluation/`: correlation, distance-error, retrieval, and plotting utilities
- `scripts/`: training, mismatch generation, retrieval, and validation-sweep entry points
- `configs/`: reproducible experiment configurations

The key architectural result is that synthetic distances and PyCBC/LALSuite matched-filter mismatch matrices are consumed through the same learning and evaluation interface.

## Completed Phases

### Phase 1: Synthetic Metric Learning

Completed a full synthetic representation-learning pipeline:

- synthetic vector generation
- pair, triplet, and distance-regression dataset utilities
- MLP encoder
- distance-regression loss
- retrieval evaluation
- correlation metrics
- plotting
- smoke tests

This phase validated the ML pipeline without gravitational-wave dependencies.

### Phase 2: Tiny GW Mismatch Integration

Integrated a tiny PyCBC/LALSuite waveform mismatch pipeline:

- nonspinning `IMRPhenomD` waveform bank
- standard PyCBC waveform generation
- standard PyCBC match calculation
- HDF5 mismatch cache
- HDF5 metadata
- GW dataset adapter
- unchanged training, retrieval, evaluation, and plotting pipeline

This phase validated that real matched-filter mismatch can replace synthetic distances without redesigning the ML code.

### Phase 3: Scientific Validation

Ran the first nontrivial validation sweep over real matched-filter mismatch banks. The sweep measured robustness across seeds, latent dimensions, MLP widths/depths, training epochs, and bank sizes.

## Experimental Configuration

Primary sweep:

```text
configs/phase3_validation.yaml
```

Base experiment:

```text
configs/waveform_bank_small.yaml
```

Waveform and mismatch settings:

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

Baseline model:

- Input dimension: 5
- Latent dimension: 4
- Hidden dimensions: `[32, 16]`
- Training epochs: 10
- Batch size: 128
- Learning rate: 0.001
- Loss: distance regression

Validation sweep:

- Seeds: `101`, `202`, `303`, `404`, `505`
- Latent dimensions: `2`, `4`, `8`, `16`, `32`, `64`
- Architectures: `[16]`, `[32, 16]`, `[64, 32]`, `[64, 32, 16]`
- Epochs: `3`, `5`, `10`, `20`
- Bank sizes: `128`, `256`, `512`, `1024`
- Retrieval K values: `5`, `10`, `20`

Physical baseline:

- Euclidean nearest-neighbor retrieval in raw `(m1, m2)` space

## Principal Quantitative Results

Seed study, mean +/- standard deviation over seeds `101`, `202`, `303`, `404`, and `505`:

| Metric | Learned latent retrieval | Raw `(m1, m2)` retrieval |
| --- | ---: | ---: |
| Recall@5 | 0.4692 +/- 0.0276 | 0.2547 +/- 0.0000 |
| Recall@10 | 0.5338 +/- 0.0253 | 0.2965 +/- 0.0000 |
| Recall@20 | 0.6736 +/- 0.0250 | 0.3924 +/- 0.0000 |
| Best recovered match@5 | 0.7086 +/- 0.0341 | 0.4492 +/- 0.0000 |
| Best recovered match@10 | 0.8398 +/- 0.0217 | 0.6055 +/- 0.0000 |
| Best recovered match@20 | 0.9258 +/- 0.0141 | 0.7422 +/- 0.0000 |

Distance-regression metrics over the same seed study:

| Metric | Mean +/- standard deviation |
| --- | ---: |
| Pearson correlation | 0.9035 +/- 0.0121 |
| Spearman correlation | 0.8999 +/- 0.0096 |
| MAE | 0.0736 +/- 0.0050 |
| RMSE | 0.0998 +/- 0.0063 |

Bank-size study:

| Bank size | Pairwise overlaps | Runtime | Learned Recall@5 | Raw `(m1, m2)` Recall@5 | Learned best@5 | Raw `(m1, m2)` best@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8,128 | 9.3 s | 0.4766 | 0.2828 | 0.6875 | 0.4219 |
| 256 | 32,640 | 28.8 s | 0.4375 | 0.2547 | 0.6758 | 0.4492 |
| 512 | 130,816 | 114.1 s | 0.4578 | 0.2375 | 0.6816 | 0.4316 |
| 1024 | 523,776 | 456.8 s | 0.6049 | 0.2301 | 0.8877 | 0.4580 |

Interpretation:

These results demonstrate that learned latent coordinates preserve matched-filter mismatch neighborhoods more effectively than naive Euclidean distance in ((m_1,m_2)) space for the tested nonspinning IMRPhenomD banks.

## Reproducibility

A clean checkout should be able to reproduce the validation with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gw.txt

python -m pytest
python -m ruff check .

python scripts/run_validation_sweep.py \
    --config configs/phase3_validation.yaml
```

Expected validation status:

- `python -m pytest`: 22 tests passing
- `python -m ruff check .`: all checks passing

Expected runtime on the current development machine:

- Full Phase 3 sweep: approximately 10 to 11 minutes
- 1024-waveform mismatch generation: approximately 7.5 minutes

Expected generated outputs:

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

Expected generated caches:

```text
data/processed/waveform_bank_128_mismatch.h5
data/processed/waveform_bank_256_mismatch.h5
data/processed/waveform_bank_512_mismatch.h5
data/processed/waveform_bank_1024_mismatch.h5
```

Generated caches and outputs are intentionally excluded from git.

## Milestone Tag

After committing the Phase 3 freeze files, create and push the milestone tag with:

```bash
git tag phase3-validation
git push origin phase3-validation
```

Do not create the tag before committing the freeze state, or the tag will point to an older repository snapshot.

## Known Limitations

- The validation uses only nonspinning compact binaries.
- The validation uses only `IMRPhenomD`.
- The mass range is deliberately narrow.
- The physical baseline is intentionally naive: raw `(m1, m2)` Euclidean distance.
- Generated banks are development-scale, not production-scale.
- The architecture, epoch, and latent-dimension sweeps are diagnostic and should not be treated as tuned model results.
- No claim is made about search acceleration.
- No claim is made about replacing matched filtering.
- Exact matched filtering remains the ground-truth verification step.

## Next Scientific Questions

The next development cycle should strengthen the physical baselines before tuning the learned model.

Immediate reviewer-proofing comparisons:

- Chirp mass
- Symmetric mass ratio
- Total mass plus mass ratio

The next question is whether the learned latent representation still outperforms physically motivated parameterizations, not just raw component masses.

Only after those baselines are implemented and measured should the project revisit architecture tuning, larger banks, additional waveform families, or search-oriented runtime studies.
