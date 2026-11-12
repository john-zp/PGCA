# Release validation

Validation date: 2026-11-12.

- The revised source was retrieved from the completed server workspace, not
  reconstructed from manuscript prose or from the initial GitHub version.
- All 53 selected source/configuration files have recorded source and release
  hashes in `source_provenance.json`. The eight root-path changes preserve every
  function and class AST. The 39-condition protocol and 36-field schema are
  retained without recomputing results.
- `tests/smoke_cpu.py` passed on the server with CUDA explicitly disabled.
  It checked the 2,828,197-parameter main graph, [B,5] threshold outputs,
  [B,1,196] attention, exact Matched/ON equivalence to PGCA-0, all ten
  Query/residual combinations, no hidden case-pathology dependency for
  Learnable/Sonographic/Constant with Residual OFF, retained matched residual
  under Shuffled Query, uniform-QK attention, finite backward gradients,
  unit-mass distributions, empty ROI handling, finite OT/path losses and
  marginal/endpoint tolerances, and the hard-prediction metric inequality.
- Synthetic checks do not validate clinical performance. No model fitting,
  optimizer step, checkpoint loading, clinical inference, or re-estimation of
  reported statistics was performed for this publication update.
- A separate CPU comparison against the untouched original server code used
  identical synthetic inputs and initialization for all ten Query/residual
  combinations. State tensors, logits, attention maps and hard predictions
  were byte-identical between the original and relocated implementations.
- The staged public tree was checked for private-data/checkpoint/image formats
  and credential patterns. All 53 released source hashes and all 39 protocol
  entries were verified. All 73 Python files in the current and legacy source
  trees parsed successfully; legacy code was not treated as the current model.
- Raw-data readers, full training drivers and clinical analysis scripts require
  private artifacts and were not run. Strong pretrained models were not
  downloaded or retrained during the release check.
- Recorded core environment: Python 3.12, torch 2.8.0+cu128, torchvision
  0.23.0+cu128, NumPy 2.3.2, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1.
  A public package installation on other systems is not an exact environment
  replay. The core smoke check does not require optional CatBoost or OpenCV.
