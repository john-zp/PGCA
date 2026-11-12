"""Numerical regression checks before interpreting new comparisons."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score
from statistics_common import hard_metrics, classification_pair, scalar_pair

rng = np.random.default_rng(12)
frame = pd.DataFrame({"image_id": [f"synthetic_{i}" for i in range(120)],
    "group_id": np.repeat(np.arange(30), 4), "y_true": rng.integers(1, 7, 120), "y_pred": rng.integers(1, 7, 120)})
value = hard_metrics(frame)
assert abs(value["Accuracy"]-accuracy_score(frame.y_true, frame.y_pred)) < 1e-12
assert abs(value["MAE"]-mean_absolute_error(frame.y_true, frame.y_pred)) < 1e-12
assert abs(value["QWK"]-cohen_kappa_score(frame.y_true, frame.y_pred, labels=np.arange(1,7), weights="quadratic")) < 1e-12
assert abs(value["Macro-F1"]-f1_score(frame.y_true, frame.y_pred, labels=np.arange(1,7), average="macro")) < 1e-12
same = classification_pair({42:frame, 3407:frame}, {42:frame.copy(), 3407:frame.copy()}, "identical", replicates=100)
assert all(r["difference"] == r["ci_low"] == r["ci_high"] == 0 for r in same)
local = frame[["image_id", "group_id"]].copy()
local["seed"], local["IoU"] = 42, rng.random(120)
same = scalar_pair(local, local.copy(), "IoU", "identical", replicates=100)
assert all(r["difference"] == r["ci_low"] == r["ci_high"] == 0 for r in same)
print("STATISTICS_TESTS_PASS")
