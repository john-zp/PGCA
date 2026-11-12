from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
from utils.io import read_feature_table, read_phrase_table


def main(feature_xlsx: str, phrase_xlsx: str, out_csv: str):
    feat = read_feature_table(feature_xlsx)
    phr = read_phrase_table(phrase_xlsx)[["id", "BI-RADS", "原始短语文本", "emb"]]
    merged = feat.merge(phr, on=["id", "BI-RADS"], how="inner")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Saved merged metadata: {out_csv}, n={len(merged)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-xlsx", required=True)
    parser.add_argument("--phrase-xlsx", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()
    main(args.feature_xlsx, args.phrase_xlsx, args.out_csv)
