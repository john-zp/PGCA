# 2026-11-12 source release

- Imported the completed server-side revised PGCA, 39-record Query protocol,
  strict Uniform-QK control, ROI supervision, strong baselines and final
  statistical/intervention analysis source.
- Retained the historical dated module layout under `revised/` to preserve
  lineage and cross-module dependencies. Eight root-path declarations now
  resolve relative to the release. Function and class ASTs were unchanged.
- Documented the actual four-stage CNN, 36 observations, case-level pathology
  embedding, independent binary ordinal head and inference inputs.
- Retained the initial code under `legacy/`; preserved the old tip on
  `archive/initial-code-20261112`. The legacy README is not the current protocol.
- Removed previously tracked clinical/sample data, generated run outputs,
  visualizations, initial result summaries and bytecode from the current tree.
  Existing Git history was not rewritten. No server originals were deleted.
- Added private-data exclusions, source hashes, reproduction boundaries and a
  CPU-only synthetic check. No new training or clinical-data evaluation was run.
- Did not publish patient-linked data, pathology text, embeddings, checkpoints,
  credentials, per-image predictions or unverified replacement generators.
