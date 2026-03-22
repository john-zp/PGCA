# BI-RADS ordinal grading baselines and PGCA

This package reorganizes the original prototype into six reproducible experiment lines:

1. **image-only**: ultrasound image -> ordinal prediction  
2. **feature-only**: structured ultrasound features -> ordinal prediction  
3. **phrase-only**: precomputed prompt / phrase embedding -> ordinal prediction  
4. **image+feature**: ultrasound image + structured features -> ordinal prediction  
5. **image+phrase**: ultrasound image + prompt embedding -> ordinal prediction  
6. **all-input**: image + structured features + prompt embedding -> ordinal prediction + optional SB/OT ROI alignment

## 1. Data layout

Put your files into the following folders.

```text
Code_v2/
├── configs/
├── data/
│   └── us/
│       ├── images/              # ultrasound images, filenames like <id>.jpg or <id>.dcm.jpg
│       ├── rois/                # ROI txt files, filenames like <id>.txt
│       ├── labels.csv           # columns: filename,label
│       ├── X_with_only_feature.xlsx
│       ├── X_with_only_phrase.xlsx
│       ├── X_with_all.xlsx
│       └── splits/
│           ├── train_ids.txt
│           ├── val_ids.txt
│           └── test_ids.txt
├── src/
├── scripts/
└── README.md
```

### What goes where

- `data/us/images/`
  - Put **all ultrasound images** here.
  - Supported names: `<id>.jpg`, `<id>.png`, `<id>.jpeg`, `<id>.dcm.jpg`
  - Example: `1.2.250....75.dcm.jpg`

- `data/us/rois/`
  - Put **expert ROI txt files** here.
  - Name each ROI file as `<id>.txt`
  - Supported ROI formats:
    - `x y w h`
    - `x,y,w,h`
    - `x1 y1 x2 y2`

- `data/us/labels.csv`
  - Two columns are expected:
    - `filename`
    - `label`
  - Labels must be one of: `2, 3, 4A, 4B, 4C, 5`

- `data/us/X_with_only_feature.xlsx`
  - Your structured feature table.
  - Must contain `id` and `BI-RADS`.
  - All remaining feature columns are treated as numeric feature inputs.

- `data/us/X_with_only_phrase.xlsx`
  - Phrase / prompt embedding table for **phrase-only**.
  - Must contain:
    - `id`
    - `BI-RADS`
    - `emb`

- `data/us/X_with_all.xlsx`
  - Prompt embedding table for **all-input**.
  - Must contain:
    - `id`
    - `BI-RADS`
    - `emb`

## 2. Important design choices in this version

- **ROI is NOT used as model input.**
  ROI is only used for:
  - optional SB/OT alignment loss during training
  - interpretability evaluation / visualization

- **Current code matches your uploaded tables.**
  In your uploaded Excel files, each sample has **one prompt embedding** in the `emb` column.  
  Therefore this implementation is a **prompt-guided** version that is fully compatible with your current data.

- **Label mapping is fixed.**
  The class mapping is:
  - `2 -> 1`
  - `3 -> 2`
  - `4A -> 3`
  - `4B -> 4`
  - `4C -> 5`
  - `5 -> 6`

## 3. Environment setup

```bash
cd Code_v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Build splits

If you already have your own `train_ids.txt`, `val_ids.txt`, `test_ids.txt`, skip this step.

```bash
python scripts/make_splits.py \
  --labels-csv data/us/labels.csv \
  --out-dir data/us/splits \
  --seed 42 \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

## 5. Optional: cache embeddings to `.npy`

This is optional. The training code can read directly from Excel.

```bash
python scripts/export_phrase_embeddings.py \
  --xlsx data/us/X_with_only_phrase.xlsx \
  --out-dir data/us/cache/phrase_only

python scripts/export_phrase_embeddings.py \
  --xlsx data/us/X_with_all.xlsx \
  --out-dir data/us/cache/all_input
```

## 6. Training commands

### A. image-only

```bash
python src/train.py --config configs/image_only.yaml
```

### B. feature-only

```bash
python src/train.py --config configs/feature_only.yaml
```

### C. phrase-only

```bash
python src/train.py --config configs/phrase_only.yaml
```

### D. image+feature

```bash
python src/train.py --config configs/image_feature.yaml
```

### E. image+phrase

```bash
python src/train.py --config configs/image_phrase.yaml
```

### F. all-input (PGCA)

```bash
python src/train.py --config configs/all_input.yaml
```


## 6b. Smoke test with included sample images

The package includes **2 sample images + ROI txt** copied from your original zip.
To quickly verify the full `all-input` pipeline, temporarily edit `configs/all_input.yaml`:

```yaml
paths:
  split_dir: data/us/sample_splits
```

Then run:

```bash
python src/train.py --config configs/all_input.yaml
```

This is only for a code sanity check, not for real experiments.

## 7. Evaluation

```bash
python src/eval.py --config configs/image_only.yaml --checkpoint runs/image_only/best.pt --split test
python src/eval.py --config configs/all_input.yaml --checkpoint runs/all_input/best.pt --split test
```

For other experiment lines, switch to the corresponding config: `image_only.yaml`, `feature_only.yaml`, `phrase_only.yaml`, `image_feature.yaml`, `image_phrase.yaml`, or `all_input.yaml`.

## 8. Notes about `all-input`

By default, `configs/all_input.yaml` uses:
- explicit structured features from `X_with_only_feature.xlsx`
- prompt embedding from `X_with_all.xlsx`
- image input from `data/us/images/`

If you want to avoid potential feature duplication inside the text prompt, you can change:

```yaml
phrase_xlsx: data/us/X_with_only_phrase.xlsx
```

inside `configs/all_input.yaml`.

## 9. Outputs

Training saves files under `runs/<experiment_name>/`:

- `best.pt`                best checkpoint by validation QWK
- `last.pt`                last checkpoint
- `metrics_epoch_*.json`   epoch metrics
- `test_predictions.csv`   generated by `eval.py`

## 10. Suggested paper wording

Because the current Excel files store a **single prompt embedding per sample**, a precise description is:

> We serialize the structured features together with the preliminary descriptive phrases into a structured prompt and encode it into one text embedding, which conditions cross-attention over image tokens.

If later you export **multiple phrase embeddings per sample**, the code can be extended to true phrase-specific attention.
