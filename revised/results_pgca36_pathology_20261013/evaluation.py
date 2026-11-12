import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, recall_score


CLASSES = np.arange(1, 7)


def metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if y.ndim != 1 or predicted.shape != y.shape or not len(y):
        raise ValueError("Nonempty, same-grain hard predictions are required")
    if not np.isin(y, CLASSES).all() or not np.isin(predicted, CLASSES).all():
        raise ValueError("Prediction or label outside explicit six-category mapping")
    accuracy, mae = accuracy_score(y, predicted), mean_absolute_error(y, predicted)
    if mae + 1e-12 < 1 - accuracy:
        raise AssertionError("MAE >= 1 - Accuracy violated for hard ordinal predictions")
    return {"Accuracy": float(accuracy), "MAE": float(mae),
            "QWK": float(cohen_kappa_score(y, predicted, labels=CLASSES, weights="quadratic")),
            "Macro-F1": float(f1_score(y, predicted, labels=CLASSES, average="macro", zero_division=0)),
            "class_recall": recall_score(y, predicted, labels=CLASSES, average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(y, predicted, labels=CLASSES).tolist(), "n": len(y)}


def threshold_calibration(y, p_ge):
    target = np.asarray(y)[:, None] >= np.arange(2, 7)[None]
    probability = np.asarray(p_ge)
    if probability.shape != target.shape or not np.isfinite(probability).all():
        raise ValueError("Invalid threshold probabilities")
    return {"threshold_Brier": float(np.mean((probability - target) ** 2)),
            "threshold_NLL": float(-np.mean(target * np.log(np.clip(probability, 1e-7, 1)) + (~target) * np.log(np.clip(1 - probability, 1e-7, 1)))),
            "nonmonotone_case_count": int(np.any(np.diff(probability, axis=1) > 1e-7, axis=1).sum()),
            "scope": "five binary threshold estimates, not a calibrated six-class distribution"}
