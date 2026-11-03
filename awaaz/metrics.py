"""Evaluation metrics. Convention everywhere: label 1 = fake, higher score = more fake."""
from __future__ import annotations

import numpy as np


def roc_points(scores: np.ndarray, labels: np.ndarray):
    """False-positive and true-positive rates at every threshold (desc)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        raise ValueError("need both real and fake samples")
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]  # one point per distinct score
    tpr = np.r_[0.0, tps[last] / pos]
    fpr = np.r_[0.0, fps[last] / neg]
    thr = np.r_[np.inf, s[last]]
    return fpr, tpr, thr


def eer(scores, labels) -> tuple[float, float]:
    """Equal error rate and the threshold where it occurs."""
    fpr, tpr, thr = roc_points(scores, labels)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    # interpolate between neighbours for a less jumpy estimate
    if i > 0 and (fnr[i] - fpr[i]) * (fnr[i - 1] - fpr[i - 1]) < 0:
        a, b = i - 1, i
        d0, d1 = fnr[a] - fpr[a], fnr[b] - fpr[b]
        w = d0 / (d0 - d1)
        e = fpr[a] + w * (fpr[b] - fpr[a])
        t = thr[b] if np.isinf(thr[a]) else thr[a] + w * (thr[b] - thr[a])
        return float(e), float(t)
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def auc(scores, labels) -> float:
    fpr, tpr, _ = roc_points(scores, labels)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def binary_report(scores, labels, threshold: float) -> dict:
    scores, labels = np.asarray(scores), np.asarray(labels)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / max(1, len(labels)), "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
