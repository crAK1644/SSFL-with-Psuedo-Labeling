"""Bookkeeping shared by every method module: not paper math, just the
generic confusion-matrix/macro-F1 helper each of FL/FD/DS-FL/SSFL's
``final_metrics``/``evaluate_*_full`` needs to build its final.json report.
"""

from __future__ import annotations

import numpy as np

__all__ = ["classification_metrics"]


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int
) -> dict:
    """Accuracy, macro-F1/precision, per-class precision/F1 and confusion matrix.

    Zero-support conventions match sklearn's ``zero_division=0``: a class
    with no predictions has precision 0, no occurrences has recall 0, and
    p + r = 0 gives F1 = 0. All scalars are plain Python floats so the dict
    (minus ``confusion_matrix``) serializes straight into final.json;
    ``confusion_matrix`` is int64 [L, L] for MetricsStore.save_confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    diag = np.diag(cm).astype(np.float64)
    pred_totals = cm.sum(axis=0).astype(np.float64)  # per predicted class
    true_totals = cm.sum(axis=1).astype(np.float64)  # per true class
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_totals > 0, diag / pred_totals, 0.0)
        recall = np.where(true_totals > 0, diag / true_totals, 0.0)
        pr = precision + recall
        f1 = np.where(pr > 0, 2 * precision * recall / np.where(pr > 0, pr, 1.0), 0.0)

    return {
        "accuracy": float(diag.sum() / max(cm.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": [float(p) for p in precision],
        "per_class_f1": [float(v) for v in f1],
        "confusion_matrix": cm,
    }
