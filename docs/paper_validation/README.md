# Paper Validation Experiments

This directory records the paper-oriented validation sequence that follows the
initial pipeline milestones. Its purpose is to preserve the scientific
validation studies intended to support the manuscript.

The earlier repository phases are separated under:

```text
docs/pipeline_milestones/
```

Those older notes document infrastructure maturation: synthetic metric
learning, tiny GW mismatch integration, and the first Phase 3 validation sweep.
The sequence below is separate and uses Phase I, Phase II, and Phase III for
paper-validation studies.

## Status

| Phase | Status | Scientific question |
| --- | --- | --- |
| Phase I: Scaling Validation | COMPLETE AND FROZEN | Does representation quality persist as waveform-bank size increases? |
| Phase II: Stronger Physical-Coordinate Baselines | COMPLETE AND FROZEN | Does the learned representation still outperform stronger fixed physical-coordinate baselines? |
| Phase III: Candidate Retrieval | PLANNED | Can latent retrieval support candidate selection for downstream exact matched filtering? |

Do not define or run the Phase III protocol from this freeze record.

## Phase Records

- [Phase I: Scaling Validation](phase_i_scaling_validation.md)
- [Phase II: Stronger Physical-Coordinate Baselines](phase_ii_stronger_physical_baselines.md)
- [Phase III: Candidate Retrieval](phase_iii_candidate_retrieval.md), planned only

## Frozen Conclusions

Phase I concluded that the learned representation continues to preserve
pairwise matched-filter mismatch relationships as bank size increases through
8192 waveforms.

At 8192 waveforms:

- Pearson: `0.9704`
- Spearman: `0.9594`
- MAE: `0.0418`
- RMSE: `0.0560`
- Recall@10: `0.7552`
- Best-match recovery@10: `0.9873`

Phase II concluded that the learned representation outperforms every tested
fixed physical coordinate system at every tested bank size and K value.

At 8192 waveforms:

- Learned latent Recall@10: `0.7552`
- Learned latent Best@10: `0.9873`
- Strongest fixed baseline `(M, eta)` Recall@10: `0.3006`
- Strongest fixed baseline `(M, eta)` Best@10: `0.6919`
- Five-feature input control Recall@10: `0.2685`
- Five-feature input control Best@10: `0.6410`

The five-feature input control is scientifically important because it shows
that the learned transformation improves matched-filter neighborhood retrieval
beyond the Euclidean geometry already present in the physical features supplied
to the encoder.

## Frozen Artifact Locations

Phase I outputs:

```text
outputs/scaling_validation/
```

Phase II outputs:

```text
outputs/phase2_physical_baselines/
```

Generated outputs are intentionally excluded from git. Their result tables are
protected by:

```text
docs/paper_validation/checksums.sha256
```

Machine-readable freeze metadata is recorded in:

```text
docs/paper_validation/freeze_manifest.json
```

Future phases must use new output directories and must not overwrite the frozen
Phase I or Phase II output directories.
