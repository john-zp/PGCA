from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def write_ids(ids, path: Path):
    path.write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")


def main(labels_csv: str, out_dir: str, seed: int, train_ratio: float, val_ratio: float):
    df = pd.read_csv(labels_csv)
    if not {"filename", "label"}.issubset(df.columns):
        raise ValueError("labels.csv must contain filename,label")
    ids = df["filename"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    temp_ratio = 1.0 - train_ratio
    ids_train, ids_temp, y_train, y_temp = train_test_split(
        ids, labels, test_size=temp_ratio, random_state=seed, stratify=labels
    )
    val_over_temp = val_ratio / temp_ratio
    ids_val, ids_test, _, _ = train_test_split(
        ids_temp, y_temp, test_size=1.0 - val_over_temp, random_state=seed, stratify=y_temp
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ids(ids_train, out_dir / "train_ids.txt")
    write_ids(ids_val, out_dir / "val_ids.txt")
    write_ids(ids_test, out_dir / "test_ids.txt")
    print(f"train={len(ids_train)} val={len(ids_val)} test={len(ids_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    main(args.labels_csv, args.out_dir, args.seed, args.train_ratio, args.val_ratio)
