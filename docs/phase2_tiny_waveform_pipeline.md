# Phase 2 Tiny Waveform Pipeline

This note freezes the tiny waveform pipeline used to validate that real matched-filter mismatch can replace synthetic distances without changing the learning, retrieval, or evaluation code.

## Purpose

The tiny pipeline is an integration test, not a scientific result. It verifies that a PyCBC/LALSuite-generated mismatch matrix can be cached and consumed through the same distance-matrix interface used by synthetic experiments.

## Waveform Setup

- Approximant: `IMRPhenomD`
- Binary type: nonspinning compact binaries
- Mass sampling: random uniform component masses with `m2 <= m1`
- Tiny mass range:
  - `m1`: 20 to 40 solar masses
  - `m2`: 10 to 30 solar masses
- Number of waveforms: 32
- Domain: frequency domain
- Sample rate: not applicable for this frequency-domain prototype
- Frequency spacing: `delta_f = 0.0625`
- Lower frequency cutoff: `f_lower = 20.0 Hz`
- Final frequency: `f_final = 512.0 Hz`
- PSD: `aLIGOZeroDetHighPower`

## Match and Mismatch

Waveforms are generated with PyCBC frequency-domain waveform tools. Pairwise matches are computed with `pycbc.filter.match` using the configured PSD and lower-frequency cutoff.

Mismatch is defined as:

```text
mismatch = 1 - match
```

The resulting matrix is symmetric, non-negative, and has a zero diagonal. Exact matched filtering remains the ground-truth comparison.

## Cached Output

The tiny run writes:

```text
data/processed/tiny_waveform_mismatch.h5
```

The HDF5 file contains:

- `features`: normalized mass-derived features used by the encoder
- `distance`: pairwise matched-filter mismatch matrix
- `mass_1`, `mass_2`: sampled component masses
- HDF5 attributes recording waveform, PSD, sampling, version, seed, and creation metadata

## Smoke Command

```bash
python scripts/train_embedding.py --config configs/waveform_bank_tiny.yaml
```

Expected runtime on the current development setup is approximately 7 seconds after PyCBC/LALSuite are installed.

The run writes:

```text
outputs/waveform_bank_tiny/distance_scatter.png
outputs/waveform_bank_tiny/retrieval_heatmap.png
outputs/waveform_bank_tiny/metrics.json
```

## Interpretation

This pipeline demonstrates infrastructure only. It shows that the same representation-learning pipeline can consume real matched-filter mismatch distances. Larger banks and baseline comparisons are required before making scientific claims about learned mismatch geometry.
