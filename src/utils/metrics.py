from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, num_classes: int = 6) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 1, num_classes)
    y_pred = np.clip(y_pred, 1, num_classes)

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t - 1, p - 1] += 1.0

    act = O.sum(axis=1)
    pred = O.sum(axis=0)
    E = np.outer(act, pred) / max(O.sum(), 1.0)
    W = np.fromfunction(lambda i, j: ((i - j) ** 2) / ((num_classes - 1) ** 2), (num_classes, num_classes))
    den = (W * E).sum()
    if den <= 0:
        return 0.0
    return float(1.0 - (W * O).sum() / den)


def compute_metrics(y_true, y_pred, num_classes: int = 6) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    mae = float(np.abs(y_true - y_pred).mean()) if len(y_true) else 0.0
    qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=num_classes)

    # macro F1 without sklearn dependency in runtime path
    f1s = []
    for cls in range(1, num_classes + 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"acc": acc, "mae": mae, "qwk": qwk, "macro_f1": macro_f1}


def save_json(obj, path: str | Path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
