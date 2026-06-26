![logo](assets/gw_mismatch_logo.png)

# gw-mismatch-learning

`gw-mismatch-learning` investigates whether the geometry induced by matched filtering can be learned as a latent representation for gravitational-wave waveforms. Rather than replacing matched filtering, this project learns coordinates in which distance approximates waveform mismatch, enabling geometry-aware search, template retrieval, and hierarchical matched-filter experiments using standard GW analysis tools.

## Scientific framing

Matched filtering defines waveform similarity through the noise-weighted overlap. Mismatch, defined as `1 - match`, induces a geometry on waveform space. This repository learns an embedding `z = f(h)` such that distances in latent space approximate the mismatch geometry.

The learned geometry is intended to support nearest-neighbor retrieval, candidate reduction, and hierarchical matched-filter search. Exact matched filtering remains the final verification step.

## What this repo is not

- Not a replacement for matched filtering.
- Not a new waveform model.
- Not a production search pipeline yet.

## Installation

For lightweight development without full gravitational-wave dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For GW waveform generation and overlap calculations, install the optional GW extras if your platform supports them:

```bash
pip install -e ".[gw]"
```

PyCBC and LALSuite installation can be platform-dependent, so the core package and smoke tests avoid importing them unless a GW-specific function is called.

## Smoke test

Run the synthetic mismatch-geometry smoke test:

```bash
python scripts/train_embedding.py --config configs/training.yaml
pytest
```

The smoke path generates mock vectors, defines a toy mismatch distance, trains a small encoder to preserve pairwise distances, and evaluates retrieval metrics.

## Planned workflow

1. Mock geometry: synthetic vectors, toy mismatch distances, learned embeddings, retrieval evaluation.
2. Standard waveform bank: PyCBC/LALSuite waveform generation, HDF5 storage, sampled match/mismatch computation.
3. Retrieval experiment: encode a bank, retrieve top-K latent candidates, and compare against brute-force exact matching.

## Design principles

- Keep GW physics standard.
- Keep ML contribution isolated and testable.
- Treat exact matched filtering as ground truth.
- Evaluate candidate reduction before claiming runtime speedup.
- Use reproducible configs for all experiments.
- Avoid committing large generated files.

## Data policy

Generated waveform banks, mismatch matrices, trained models, and experiment outputs are intentionally excluded from git. Use `data/`, `models/`, and `outputs/` for local artifacts.
