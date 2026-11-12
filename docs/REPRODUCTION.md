# Reproduction guide

## Safe public checks

Run `python tests/smoke_cpu.py` from the repository root. It uses no clinical
cache and performs no optimizer steps. It exercises all Query/residual interfaces
and the released spatial-loss functions with synthetic inputs.

The base graph uses 36 observations and a 512-vector, a four-stage CNN, four
attention heads, and five independent binary ordinal thresholds. The pretrained
ResNet-50/DenseNet-121 alternatives exist only in the strong-baseline module.

## Frozen training protocol

`revised/results_pgca36_pathology_20261013/common_config.json` preserves the
historical shared configuration: 224 pixels, batch 16, AdamW, learning rate
0.0003, weight decay 0.0001, 30 epochs, gradient clipping 5.0, and seeds
42/3407/2026. Checkpoint selection maximizes validation QWK, then minimizes
validation MAE, keeping the earliest tied epoch.

`revised/query_mechanism_20261102/protocol.json` preserves the bounded 39-record
matrix. Stage A and B have no spatial term. Stage C uses ROI-KL weight 0.01
with the pathology residual disabled. Generic OT/permutation/bootstrap fields
in the inherited base configuration do not activate those mechanisms in the
Query trainer; inspect the condition and `objective` actually called.

For the historical spatial extensions, static OT uses the recorded outer weight
0.2 and internal cost multiplier 0.3; Full adds the path penalty at 0.05. The
ROI-KL weight selected by validation was 0.01 for ROI-KL alone and 0.05 for
Static OT + ROI-KL, as recorded in the original `selection_lock.json`. The
candidates were 0.01, 0.05 and 0.2. These are retained settings, not a new
test-set search; the complete original selection record remains private.

## Reference execution entry points

These are **data- and provenance-dependent reference programs**, not commands
to run on a fresh public clone:

| Program | Role |
| --- | --- |
| `results_pgca36_pathology_20261013/train_one.py --model PGCA-0 --seed 42` | Base PGCA and modality benchmarks |
| `results_pgca36_pathology_20261013/mechanism_validation_20261014/run_strict_ablation.py` | Uniform-QK strict mechanism control |
| `query_mechanism_20261102/train_query.py B_Matched_OFF_seed42` | One final Query/residual condition |
| `roi_supervision_20261021/train_roi.py` | ROI-KL / Static OT + ROI-KL reference fits |
| `supplementary_validation_20261014/run_strong.py` | Validation-selected CatBoost and pretrained visual baselines |
| `final_submission/source/analysis/recompute_statistics.py` | Paired group-bootstrap and group-swap/Holm analysis |
| `final_submission/source/analysis/query_interventions.py` | Frozen model, Query-only interventions; not training |

Paths in the table are relative to `revised/`. Invoke each program in its own
process. Some preparation programs read the institution's original Excel and
linkage records; those inputs are intentionally absent here.

## Locks are not fabricated

The reference training drivers require the private dataset, donor maps, original
environment, source locks and preflight records. They stop if these are absent
or disagree. The public package relocates eight top-level root declarations,
so an old byte-level source lock is not a valid lock for this relocated release.
Do not edit old lock digests to make a check pass. Exact historical execution
uses the original frozen workspace. A new authorized reproduction requires a
separate workspace, a new source/environment lock and independently completed
preflight checks, while retaining the original split and protocol. This code
publication did not perform such a new clinical reproduction.

## Statistical and interpretation boundaries

All four classification metrics are derived from the same hard predictions.
`evaluation.metrics` checks the necessary MAE >= 1 - Accuracy condition. The
five sigmoid outputs need not be monotone and are not six-class probabilities.
The final intervention script defines strict adjacent-threshold violations;
the older calibration helper's tolerance-based count is a separate diagnostic.

Final paired risk-group analyses use the 555 test images / 380 groups and, for
localization, the common 383 valid-ROI images / 290 groups. The final 12 primary
classification tests use Holm adjustment; bootstrap intervals and adjusted
permutation tests are different outputs. The earlier analysis stages retain
their own historical protocols and must not substitute for the final analysis.

ROI-KL, static OT and finite-state path supervision are distinct objectives.
Threshold IoU@0.5, fixed-area IoU and Pointing Game are different localization
measurements. Rectangular-ROI correspondence is not causal explanation fidelity.
The repeated-use test set is not external confirmatory validation.
