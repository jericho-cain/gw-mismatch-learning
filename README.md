![gw-mismatch-learning logo](assets/gw_mismatch_logo.png)

# Learning the Geometry of Matched-Filter Mismatch

This repository contains the code and reproducibility record for the manuscript
*Learning the Geometry of Matched-Filter Mismatch for Gravitational-Wave Template
Banks with Machine Learning* by Jericho Cain.

Matched filtering defines waveform similarity through the noise-weighted overlap.
This project uses the exact pairwise matched-filter mismatch (PMM) between waveforms as
a supervision signal for learning a low-dimensional coordinate map. Euclidean distance
in the learned coordinates approximates PMM, allowing nearest-neighbor retrieval to
select a small candidate set for subsequent exact matched-filter evaluation.

The learned representation does **not** replace matched filtering. Exact matched
filtering remains the final scoring operation throughout the study.

## Principal results

The paper evaluates nonspinning `IMRPhenomD` waveform banks containing up to 8192
templates. Within the waveform family and parameter range studied:

- The learned coordinates preserve global pairwise mismatch relationships and local
  matched-filter neighborhoods.
- Learned retrieval outperforms Euclidean retrieval in all five fixed physical
  coordinate systems considered, including the complete five-feature encoder input.
- For 512 previously unseen query waveforms and an 8192-template bank, 10 retrieved
  candidates recover the exhaustive best-matching template for 99.02% of queries.
- Twenty candidates recover the exhaustive best match for all 512 queries in the
  reported experiment, reducing exact matched-filter evaluations by a factor of 409.6.
- Neighborhood retrieval is close to saturation by approximately four latent
  dimensions, while global distance preservation continues to improve at higher
  dimensions.
- Classical Multidimensional Scaling provides a strong deterministic in-sample
  reference, while the learned encoder additionally supplies an explicit mapping for
  previously unseen waveforms.

These are candidate-reduction and representation-learning results, not measurements of
end-to-end gravitational-wave search acceleration.

## Scope and limitations

The experiments are restricted to nonspinning `IMRPhenomD` waveforms sampled from the
mass ranges defined in the committed configurations. The encoder inputs are engineered
mass-derived features, and out-of-sample queries come from the same waveform family and
parameter distribution as the training bank. The study does not establish performance
for aligned spin, precession, eccentricity, other waveform approximants, or production
search pipelines.

Construction of the complete PMM matrix scales quadratically with bank size and is the
dominant offline cost. The learned representation is used only for candidate selection;
exact matched filtering is retained for final verification.

## Reproducing the paper

Use the annotated Git tag:

```bash
git checkout v1.0.0-paper
```

The complete ordered reproduction procedure is in
[`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md). It covers:

1. Environment creation and dependency installation.
2. Waveform-bank scaling and pairwise mismatch construction.
3. Physical-coordinate baselines.
4. Out-of-sample candidate retrieval.
5. The five-seed latent-dimension sweep.
6. The deterministic Classical MDS reference.
7. Tests, linting, and frozen-result checksum verification.

The 8192-template PMM calculation is the dominant computation and took approximately
2.4 hours on the machine used for the paper. Runtime will vary by hardware and software
environment.

Detailed definitions and validation records for individual experiments are maintained
under [`docs/paper_validation/`](docs/paper_validation/README.md).

## Installation

Python 3.10 or newer is required. For the gravitational-wave experiments, use an
isolated environment because PyCBC and LALSuite constrain parts of the scientific
Python stack:

```bash
python -m venv .venv-gw
source .venv-gw/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gw,dev]"
```

The requirements-file equivalent is:

```bash
python -m pip install -r requirements-gw.txt
```

Confirm the environment with:

```bash
python -m pytest
python -m ruff check .
```

For lightweight development without PyCBC or LALSuite:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/train_embedding.py --config configs/training.yaml
```

## Repository structure

```text
configs/                 Committed experiment definitions
data/                    Generated waveform banks and mismatch caches (not tracked)
docs/paper_validation/   Detailed validation and frozen-result records
models/                  Generated model checkpoints (not tracked)
notebooks/               Demonstration notebooks
outputs/                 Generated experiment outputs (not tracked)
scripts/                 Experiment entry points
src/                     Reusable package implementation
tests/                   Unit and integration tests
```

Large waveform banks, mismatch matrices, checkpoints, and generated figures are
excluded from Git. Compact validation records, configurations, checksums, and the code
needed to regenerate the paper experiments are committed. Frozen Phase I--III tables
can be checked with:

```bash
sha256sum --check docs/paper_validation/checksums.sha256
```

after the corresponding outputs have been recreated.

## Citation

The manuscript citation and arXiv identifier will be added when the preprint becomes
public. Until then, please cite the repository and manuscript title:

```text
Jericho Cain, "Learning the Geometry of Matched-Filter Mismatch for
Gravitational-Wave Template Banks with Machine Learning," manuscript in preparation.
```

## License

This project is released under the [MIT License](LICENSE).
