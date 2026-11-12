# Private data contract

No clinical data are included in this release. Access requires the appropriate
institutional and ethical approvals. Do not commit clinical images, pathology
text, cached embeddings, patient/lesion identifiers, private manifests or
per-image predictions to this repository.

The baseline `Cohort` reader expects the following authorized cache under
`revised/results_pgca36_pathology_20261013/data/`:

| File | Content |
| --- | --- |
| `images.npy` | Grayscale images, N x 1 x 224 x 224; historical normalized float16 cache cast to float32 on read |
| `roi_masks.npy` | Corresponding rectangular ROI masks, N x 1 x 224 x 224; empty ROI cases retained for classification |
| `features36.npy` | N x 36 binary observations in the exact `feature_schema.json` order |
| `pathology_embeddings.npy` | N x 512 matched case-level stored embeddings |
| `y.npy` | Integer labels 1..6 in BI-RADS order 2, 3, 4A, 4B, 4C, 5 |
| `dataset_manifest.csv` | Ordered linkage and frozen split; required reader fields include image_id, patient_key, group_id and split |

The data preparation/audit code requires additional provenance columns such as
array_index, y, embedding_sha256, source_form and source_excel_row. Those records
remain private. All arrays must use exactly the manifest row order. Never infer
patient identities from an arbitrary filename or rebuild a favorable test split.

The 36 field names and order are public in `feature_schema.json`. They exclude
the two diagnostic-nature columns from the old 38-dimensional input. Missing
observations were isolated, not silently filled with zero. These are recorded
sonographic observations, not the old 20-dimensional ROI-quantification scheme.

The final Query controls additionally require private
`query_mechanism_20261102/design/query_donors.npy` and
`design/train_mean_embedding.npy`. The donor array is aligned with the full
cohort: donors stay within each split, outside the recipient risk group, and are
shared by images in the same recipient group. The training-only embedding mean
must not include validation/test cases. Preserve the archived mapping and
ordering; changing identifiers can change deterministic donor/augmentation rules.

The original text field contains pathology diagnoses. This release uses the
stored embeddings and does not offer an unverified replacement Qwen generator.
An exactly equivalent generator/tokenizer/pooling reconstruction was unavailable;
regenerated embeddings must not silently be presented as the frozen inputs.
