# Paper Validation Experiments

This directory records the paper-oriented validation sequence that follows the
initial pipeline milestones.

The earlier repository phases are now separated under:

```text
docs/pipeline_milestones/
```

Those files document infrastructure maturation: synthetic metric learning, tiny
GW mismatch integration, and the first Phase 3 validation sweep. The documents
in this directory use a new sequence for scientific validation studies intended
to support the manuscript.

## Experiment Sequence

| Phase | Status | Question |
| --- | --- | --- |
| Phase I | Complete | Does representation quality persist as waveform-bank size increases? |
| Phase II | Planned | Does the learned representation still outperform stronger physical-coordinate baselines? |
| Phase III | Planned | Can latent retrieval support candidate selection for downstream exact matched filtering? |

## Records

- [Phase I: Scaling Validation](phase_i_scaling_validation.md)
- Phase II: Stronger Physical Baselines, planned
- Phase III: Candidate Retrieval, planned

Generated outputs are intentionally excluded from git. Local artifacts for
Phase I are written under:

```text
outputs/scaling_validation/
```
