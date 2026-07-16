# Principal k=8 Phase III candidate retrieval

Complete. Query masses and exhaustive best-match references were reused from the frozen
Phase III per-query output. Waveforms were regenerated deterministically because no
waveform cache was persisted; exhaustive query-template matching was not rerun.

The smallest K reaching at least 99% exact-best recovery is K=10 (0.99023). The smallest
K reaching 100% is K=50. Maximum match penalties at K=5, 10, and 20 are 9.586e-4,
9.030e-5, and 7.451e-6. Learned retrieval takes 0.0254 s over all 512 queries; total
candidate-pipeline times at K=5, 10, and 20 are 0.684, 1.340, and 2.654 s.
