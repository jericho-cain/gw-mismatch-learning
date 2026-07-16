# Latent-Dimension Robustness Sweep

## Status

Implementation complete; full experiment not yet launched. This note must be updated with
the generated numerical findings only after review of `outputs/latent_dimension_sweep/`.

## Design

The experiment reuses `data/processed/waveform_bank_8192_mismatch.h5` and refuses to
generate a replacement. Latent dimensions 2, 3, 4, 8, and 16 are trained with seeds
1234, 2345, 3456, 4567, and 5678. Seed 1234 is the frozen Phase I seed. Architecture,
optimizer, pair sampling, pair count, training schedule, and full-bank evaluation are
inherited from `configs/waveform_bank_small.yaml`; only the latent output width and the
explicit run seed vary.

Phase I used a single global seed for model initialization, pair sampling, and shuffled
training batches, so this experiment preserves that policy. Phase I had no held-out
validation split. Consequently `validation_loss` is recorded as unavailable rather than
introducing a new protocol.

For each metric, the aggregate uses the arithmetic mean, sample standard deviation
(`ddof=1`), and a two-sided 95% Student-t interval with four degrees of freedom. Failed or
missing runs prevent aggregation.

Run with:

```bash
python scripts/run_robustness_experiments.py \
  --config configs/latent_dimension_sweep.yaml
```

