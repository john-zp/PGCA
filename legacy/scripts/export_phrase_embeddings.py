from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
from utils.io import read_phrase_table


def main(xlsx: str, out_dir: str):
    df = read_phrase_table(xlsx)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        sid = str(row["id"])
        emb = row["emb_array"].astype(np.float32)
        np.save(out_dir / f"{sid}.npy", emb)
    print(f"Saved {len(df)} embeddings to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args.xlsx, args.out_dir)
