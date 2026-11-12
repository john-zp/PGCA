# Pathology-guided Cross-modal Attention (PGCA)

Research code for six-level BI-RADS prediction from breast ultrasound images,
36 structured sonographic observations, and a stored case-level pathology-text
embedding. The target order is **2, 3, 4A, 4B, 4C, 5**, encoded as **1 through 6**.

## Current implementation

The `revised/` directory contains the implementation used in the revised study,
retrieved from the completed research workspace on 2026-11-12. The initial
implementation is retained under `legacy/` for historical inspection, not for
reproducing the revised experiments. The previous repository tip is also
preserved on `archive/initial-code-20261112`.

- Four-stage **non-residual CNN**, grayscale 224 x 224 images, 14 x 14 tokens.
- A 512-dimensional case-level pathology embedding projected to 256 dimensions.
- Four-head cross-attention; image-global, attended-text and 36-field feature
  representations are fused for prediction.
- Five independent binary thresholds, with grade `1 + sum(sigmoid(logits) >= 0.5)`.
  These outputs are not a normalized six-class distribution and have no imposed
  monotonicity constraint.
- Controlled Matched, Shuffled, Learnable, Sonographic and Constant Query sources,
  with explicit pathology Residual ON/OFF settings.
- Training-only rectangular-ROI supervision: ROI-KL, static Sinkhorn OT,
  Static OT + ROI-KL, and a finite-state bridge/path penalty.

Pathology embeddings are inputs at both training and inference for the matched
PGCA model. ROI is not a prediction input. This code does not establish use at an
ultrasound-only, pre-pathology clinical decision point, and is not a clinical device.
There is no teacher-student or distillation branch.

## Where to start

| Directory under `revised/` | Purpose |
| --- | --- |
| `results_pgca36_pathology_20261013/` | Revised PGCA factory, simple modality baselines, ordinal head, spatial losses, hard-prediction metrics and reference training driver |
| `results_pgca36_pathology_20261013/mechanism_validation_20261014/` | Uniform-QK strict control and parameter randomization |
| `query_mechanism_20261102/` | Final 39-condition-seed matrix, group-consistent Query sources, residual controls and equal-ROI-KL comparisons |
| `roi_supervision_20261021/` | ROI-KL and Static OT + ROI-KL reference experiments |
| `supplementary_validation_20261014/` | Strong image/CatBoost baselines, earlier semantic controls and shared localization/statistics utilities |
| `final_submission/source/analysis/` | Final group-bootstrap, group-swap/Holm and frozen-Query intervention analysis code |
| `audit_20261013/reference_B/src/` | Frozen topology dependency, used through the revised model factory |

The dated directory names preserve the experiment lineage and import layout;
`supplementary_validation` is a historical code-directory name, not a separate
current dataset. Run each reference program in a separate Python process because
the historical stages contain different modules named `query_model.py`.

## Installation and data-free check

Python 3.12 was used in the frozen environment. Install PyTorch and torchvision
for your CPU/CUDA platform, then install the remaining dependencies:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0
python -m pip install -r requirements.txt
python tests/smoke_cpu.py
```

`requirements-optional.txt` records the CatBoost and OpenCV versions used by
the strong-baseline and original-data preparation programs, respectively.

The smoke check uses synthetic inputs on CPU only. It checks dimensions,
residual isolation, uniform attention, ROI-KL/OT/path gradients and the metric
contract. It does not load clinical records, download pretrained weights, run
an optimizer or reproduce any reported performance value.

Build the main prediction model directly from the revised factory:

```python
import sys
sys.path.insert(0, "revised/results_pgca36_pathology_20261013")
from experiment_models import make_model

model = make_model("PGCA-0")
# out = model(image, features36, pathology_embedding)
# image: [B, 1, 224, 224]; features36: [B, 36]; embedding: [B, 512]
# No ROI argument is required by the prediction network.
```

Use `make_model`, not the historical `PGCAModel` constructor directly. The factory
replaces its legacy head with `IndependentOrdinalHead`. The retained class name
`ResNetTokenEncoder` refers to the four-stage CNN above, **not** a ResNet. These
names are retained to avoid silently changing frozen construction/state behavior.

## Reproduction boundaries

The frozen cohort contains 3,709 images with train/validation/test counts
2,599/555/555. Seeds are 42, 3407 and 2026. The final Query matrix comprises
15 Residual-ON/no-spatial-loss records, 15 Residual-OFF records, and 9
Residual-OFF/equal-ROI-KL records. Three Matched/ON checkpoints were historically
reused; 36 new fits produced the remaining records. These are image counts and
model-seed records, not numbers of independent patients.

Clinical images, reports, embeddings, patient-linked splits, predictions and
checkpoints are **not distributed** by this source release. See
[the private-data contract](docs/DATA_CONTRACT.md) and
[reproduction instructions](docs/REPRODUCTION.md). The training drivers retain
their hash/preflight gates: a fresh public clone is not a one-command reproduction
of the clinical experiments. Missing private provenance must not be bypassed by
fabricating successful checks or replacing the frozen split.

The final analysis scripts, rather than the initial result summaries, identify
the relevant paired comparisons and the 12-test Holm family. Earlier semantic
and whole-text controls must not be relabeled as the final group-consistent
Query-only experiments. Neither favorable point estimates nor ROI overlap imply
universal statistical superiority or causal explanation fidelity.

## Source preservation and privacy

Model, loss, metric and training function/class bodies are retained from the
server snapshots. Eight module-level project-root declarations were made
relative for this source release; no architecture or loss was redesigned.
[Source provenance](docs/source_provenance.json) records the source and released
file hashes. [The release notes](CHANGELOG.md) describe the migration.

Old sample data and run artifacts were removed from the current public tree.
This is not a rewrite or purge of previously published Git history. Historical
server code, checkpoints and original data were not overwritten.
