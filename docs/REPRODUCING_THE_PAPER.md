# Reproducing the Paper Experiments

This is the canonical entry point for reproducing the numerical experiments in
*Learning the Geometry of Matched-Filter Mismatch for Gravitational-Wave Template
Banks with Machine Learning*. Run all commands from the repository root at the
`v1.0.0-paper` tag.

The manuscript uses nonspinning `IMRPhenomD` waveforms, PyCBC/LALSuite matched
filtering, and the configuration files committed with the paper snapshot. Generated
waveform banks, mismatch matrices, models, and outputs are excluded from Git because
of their size; the commands below recreate them.

## 1. Environment

Python 3.10 or newer is required. Use a clean environment for the gravitational-wave
dependencies:

```bash
python -m venv .venv-gw
source .venv-gw/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gw,dev]"
```

Confirm the installation before starting the long calculations:

```bash
python -m pytest
python -m ruff check .
```

PyCBC and LALSuite availability can depend on platform. The exact dependency ranges
used by the project are recorded in `pyproject.toml` and `requirements-gw.txt`.

## 2. Reproduction order

The stages are intentionally ordered because later experiments reuse the frozen
8192-waveform cache and learned results produced by earlier stages. Do not run stages
that depend on a cache before the scaling study has created it.

### A. Scaling study (Figures 2--3 and offline runtimes)

```bash
python scripts/run_scaling_validation.py \
  --config configs/scaling_validation.yaml
```

This creates waveform banks and complete pairwise mismatch matrices for
`N = 128, 256, 512, 1024, 2048, 4096, 8192`, trains the principal `k=4`, seed 1234
encoder at each size, and evaluates global geometry and in-sample retrieval. Outputs
are written to:

```text
data/processed/waveform_bank_<N>_mismatch.h5
outputs/scaling_validation/
```

The complete 8192-by-8192 mismatch construction is the dominant computation. The
paper's measured wall-clock values are observations from the original machine, not
portable runtime expectations.

### B. Physical-coordinate baselines (Figure 4)

Requires the scaling-study caches and learned outputs.

```bash
python scripts/run_phase2_physical_baselines.py \
  --config configs/phase2_physical_baselines.yaml
```

This evaluates the learned representation and all five standardized Euclidean
physical-coordinate baselines on the same banks. Outputs are written to:

```text
outputs/phase2_physical_baselines/
```

### C. Out-of-sample candidate retrieval (Figures 5--6 and Tables 3 and 5)

Requires the 8192-waveform cache and trained principal encoder from the scaling study.

```bash
python scripts/run_phase3_candidate_retrieval.py \
  --config configs/phase3_candidate_retrieval.yaml
```

This generates the fixed 512-query set, performs learned and physical-coordinate
candidate retrieval, and compares each candidate set with exhaustive exact matched
filtering. Outputs are written to:

```text
outputs/phase3_candidate_retrieval/
```

### D. Latent-dimension, five-seed sweep (Figure 7 and Table 6)

Requires `data/processed/waveform_bank_8192_mismatch.h5` from stage A. The runner
validates completed runs before skipping them, so `--resume` is safe after an
interruption. Aggregation requires exactly five valid runs per dimension.

```bash
python scripts/run_robustness_experiments.py \
  --config configs/latent_dimension_sweep.yaml \
  --resume
```

This trains dimensions `k = 2, 3, 4, 8, 16` with seeds 1234, 2345, 3456, 4567,
and 5678. Means and two-sided 95% Student-t confidence intervals are written to:

```text
outputs/latent_dimension_sweep/aggregate.csv
outputs/latent_dimension_sweep/runs.csv
outputs/latent_dimension_sweep/summary.json
outputs/latent_dimension_sweep/figures/
```

### E. Classical MDS reference (Figure 8 and Table 7)

Requires the 8192-waveform cache from stage A and the completed dimension sweep from
stage D. Validate the eigensolver implementation on the smaller cached banks before
running the full deterministic reference:

```bash
python scripts/run_mds_baseline.py \
  --config configs/mds_baseline.yaml \
  --validate-only

python scripts/run_mds_baseline.py \
  --config configs/mds_baseline.yaml
```

Outputs are written to:

```text
outputs/mds_baseline/metrics_by_dimension.csv
outputs/mds_baseline/comparisons.csv
outputs/mds_baseline/eigenspectrum.json
outputs/mds_baseline/summary.json
outputs/mds_baseline/figures/
```

Classical MDS is deterministic and in-sample. It has no out-of-sample Phase III result
in this repository.

## 3. Verification

After all stages complete, run the test and lint suites again:

```bash
python -m pytest
python -m ruff check .
```

The frozen Phase I--III result tables can be checked against the committed checksum
record after their output directories have been recreated:

```bash
sha256sum --check docs/paper_validation/checksums.sha256
```

Detailed experiment definitions, frozen findings, output schemas, and scientific
scope are recorded under [`docs/paper_validation/`](paper_validation/README.md).

## 4. Optional validation studies not required by the manuscript

The following studies are preserved for additional robustness checks but are not part
of the ordered manuscript reproduction above:

```bash
python scripts/run_robustness_experiments.py \
  --config configs/seed_robustness.yaml \
  --resume

python scripts/run_principal_k8.py \
  --config configs/principal_k8.yaml \
  --resume
```

They write only to `outputs/seed_robustness/` and `outputs/principal_k8/`, respectively,
and do not overwrite the paper's frozen principal `k=4` results.

