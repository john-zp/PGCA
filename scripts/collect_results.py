from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_META = {
    "image_only": {
        "Model": "Image-only",
        "Image": "✓",
        "Feature": "-",
        "Phrase": "-",
    },
    "feature_only": {
        "Model": "Feature-only",
        "Image": "-",
        "Feature": "✓",
        "Phrase": "-",
    },
    "phrase_only": {
        "Model": "Phrase-only",
        "Image": "-",
        "Feature": "-",
        "Phrase": "✓",
    },
    "image_feature": {
        "Model": "Image+Feature",
        "Image": "✓",
        "Feature": "✓",
        "Phrase": "-",
    },
    "image_phrase": {
        "Model": "Image+Phrase",
        "Image": "✓",
        "Feature": "-",
        "Phrase": "✓",
    },
    "all_input": {
        "Model": "All-input",
        "Image": "✓",
        "Feature": "✓",
        "Phrase": "✓",
    },
}


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(
    runs_dir: str = "runs",
    split: str = "test",
    out_csv: str = "results_summary.csv",
    out_tex: str = "results_summary.tex",
):
    runs_dir = Path(runs_dir)
    rows = []

    for run_name, meta in MODEL_META.items():
        # 如果你的 eval.py 保存的是 metrics_test.json，就保持这一行不变
        # metrics_path = runs_dir / run_name / f"metrics_{split}.json"

        # 如果你的 eval.py 保存的是 test_metrics.json，把上面一行改成：
        metrics_path = runs_dir / run_name / f"{split}_metrics.json"

        if not metrics_path.exists():
            print(f"[WARN] Missing metrics file: {metrics_path}")
            continue

        metrics = load_metrics(metrics_path)

        row = {
            **meta,
            "Acc": safe_float(metrics.get("acc")),
            "MAE": safe_float(metrics.get("mae")),
            "QWK": safe_float(metrics.get("qwk")),
            "Macro-F1": safe_float(metrics.get("macro_f1")),
        }
        rows.append(row)

    if not rows:
        print("[ERROR] No metrics files found. Please check runs directory and metric filenames.")
        return

    df = pd.DataFrame(
        rows,
        columns=["Model", "Image", "Feature", "Phrase", "Acc", "MAE", "QWK", "Macro-F1"],
    )

    # 四位小数
    for col in ["Acc", "MAE", "QWK", "Macro-F1"]:
        df[col] = df[col].map(lambda x: round(x, 4) if pd.notna(x) else x)

    # 保存 CSV
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 生成 LaTeX 表格
    latex_table = df.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
    )
    Path(out_tex).write_text(latex_table, encoding="utf-8")

    print(df)
    print(f"\nSaved CSV to: {out_csv}")
    print(f"Saved LaTeX table to: {out_tex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-csv", default="results_summary.csv")
    parser.add_argument("--out-tex", default="results_summary.tex")
    args = parser.parse_args()

    main(
        runs_dir=args.runs_dir,
        split=args.split,
        out_csv=args.out_csv,
        out_tex=args.out_tex,
    )