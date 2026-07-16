# Seed Robustness Across Bank Size

## Status

Implementation complete; full experiment not yet launched. This note must be updated with
the generated numerical findings only after review of `outputs/seed_robustness/`.

## Design

The experiment reuses the seven frozen mismatch-matrix caches for bank sizes 128, 256,
512, 1024, 2048, 4096, and 8192. It refuses to run if a cache is absent and never invokes
waveform or mismatch generation. Every size uses latent dimension 4 and seeds 1234, 2345,
3456, 4567, and 5678. Seed 1234 reproduces the original Phase I seed configuration.

The frozen implementation uses one global seed for model initialization, uniform pair
sampling with replacement, and shuffled training batches; no separate train/validation
split exists. That behavior is retained. Physical-coordinate metrics are deterministic
for a fixed frozen bank and are treated as reference values, without uncertainty bands.

For each learned metric, the aggregate uses the arithmetic mean, sample standard deviation
(`ddof=1`), and a two-sided 95% Student-t interval with four degrees of freedom. Failed or
missing runs prevent aggregation.

Run with:

```bash
python scripts/run_robustness_experiments.py \
  --config configs/seed_robustness.yaml
```
