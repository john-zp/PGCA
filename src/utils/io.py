from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

LABEL_TO_INT = {
    "2": 1,
    "3": 2,
    "4A": 3,
    "4B": 4,
    "4C": 5,
    "5": 6,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
VALID_LABELS = list(LABEL_TO_INT.keys())


def normalize_label(label: str) -> str:
    s = str(label).strip().upper()
    s = s.replace("BI-RADS", "").replace("BIRADS", "").replace("BI RADS", "")
    s = s.replace(":", " ").replace("-", " ").replace(" ", "")
    if s not in LABEL_TO_INT:
        raise ValueError(f"Unsupported label: {label}")
    return s


def label_to_int(label: str) -> int:
    return LABEL_TO_INT[normalize_label(label)]


def load_labels_csv(path: str | Path) -> Dict[str, int]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        labels = {}
        for row in reader:
            sid = row.get("filename") or row.get("﻿filename") or row.get("id") or row.get("stem")
            if sid is None:
                raise ValueError("labels.csv must contain a filename/id column")
            labels[str(sid).strip()] = label_to_int(row["label"])
    return labels


def parse_embedding_cell(value, default_dim: int = 512) -> np.ndarray:
    if value is None:
        return np.zeros(default_dim, dtype=np.float32)

    try:
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float32)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float32)
        elif isinstance(value, str):
            s = value.strip()
            if s == "" or s == "[]" or s.lower() == "nan":
                return np.zeros(default_dim, dtype=np.float32)
            arr = np.asarray(ast.literal_eval(s), dtype=np.float32)
        else:
            return np.zeros(default_dim, dtype=np.float32)
    except Exception:
        return np.zeros(default_dim, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(default_dim, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.shape[0] != default_dim:
        if arr.shape[0] > default_dim:
            arr = arr[:default_dim]
        else:
            pad = np.zeros(default_dim - arr.shape[0], dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
    return arr.astype(np.float32)


def read_feature_table(xlsx_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    if "id" not in df.columns:
        raise ValueError(f"{xlsx_path} must contain an 'id' column")
    if "BI-RADS" not in df.columns:
        raise ValueError(f"{xlsx_path} must contain a 'BI-RADS' column")
    keep = [c for c in df.columns if c not in ["图片子目录名称"]]
    df = df[keep].copy()
    df["BI-RADS"] = df["BI-RADS"].astype(str).str.strip().str.upper()
    df = df[df["BI-RADS"].isin(VALID_LABELS)].copy()
    feature_cols = [c for c in df.columns if c not in ["id", "BI-RADS", "原始短语文本", "emb"]]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["id"] = df["id"].astype(str)
    df["label_int"] = df["BI-RADS"].map(label_to_int)
    return df


def read_phrase_table(xlsx_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    required = {"id", "BI-RADS", "emb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{xlsx_path} missing columns: {sorted(missing)}")
    df = df.copy()
    if "原始短语文本" not in df.columns:
        df["原始短语文本"] = ""
    df["id"] = df["id"].astype(str)
    df["BI-RADS"] = df["BI-RADS"].astype(str).str.strip().str.upper()
    df = df[df["BI-RADS"].isin(VALID_LABELS)].copy()
    df["label_int"] = df["BI-RADS"].map(label_to_int)
    df["emb_array"] = df["emb"].apply(parse_embedding_cell)
    return df


def read_ids(path: str | Path) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image_path(images_dir: str | Path, sample_id: str) -> Optional[Path]:
    images_dir = Path(images_dir)
    candidates = [
        images_dir / f"{sample_id}.jpg",
        images_dir / f"{sample_id}.png",
        images_dir / f"{sample_id}.jpeg",
        images_dir / f"{sample_id}.dcm.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback glob
    matches = list(images_dir.glob(f"{sample_id}*"))
    for p in matches:
        if p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            return p
    return None


def load_grayscale_image(path: str | Path, image_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return img


def parse_roi_txt(path: str | Path) -> Optional[Tuple[float, float, float, float]]:
    path = Path(path)
    if not path.exists():
        return None
    s = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) < 4:
        return None
    x1, y1, a, b = map(float, nums[:4])
    if a > x1 and b > y1:
        w, h = a - x1, b - y1
    else:
        w, h = a, b
    if w <= 0 or h <= 0:
        return None
    return x1, y1, w, h


def find_roi_path(rois_dir: str | Path, images_dir: str | Path, sample_id: str) -> Optional[Path]:
    rois_dir = Path(rois_dir)
    images_dir = Path(images_dir)
    candidates = [rois_dir / f"{sample_id}.txt", images_dir / f"{sample_id}.txt"]
    for p in candidates:
        if p.exists():
            return p
    return None


def make_resized_roi_mask(
    roi_xywh: Tuple[float, float, float, float],
    original_hw: Tuple[int, int],
    target_size: int,
) -> np.ndarray:
    orig_h, orig_w = original_hw
    x, y, w, h = roi_xywh
    sx = target_size / max(orig_w, 1)
    sy = target_size / max(orig_h, 1)
    x1 = int(round(x * sx))
    y1 = int(round(y * sy))
    x2 = int(round((x + w) * sx))
    y2 = int(round((y + h) * sy))
    x1 = max(0, min(target_size - 1, x1))
    y1 = max(0, min(target_size - 1, y1))
    x2 = max(x1 + 1, min(target_size, x2))
    y2 = max(y1 + 1, min(target_size, y2))
    mask = np.zeros((target_size, target_size), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask
