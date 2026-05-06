
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def _roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    uniq, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sum_ranks = np.bincount(inv, weights=ranks, minlength=len(uniq))
        avg_ranks = sum_ranks / counts
        ranks = avg_ranks[inv]

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def _average_precision_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos_total = int((y_true == 1).sum())
    if pos_total == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(pos_total)

    ap = 0.0
    prev_recall = 0.0
    for p, r, yt in zip(precision, recall, y_true):
        if yt == 1:
            ap += p * max(0.0, r - prev_recall)
            prev_recall = r
    return float(ap)


def binary_curve_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    out = {"auroc": float("nan"), "auprc": float("nan")}
    if y_true.size == 0:
        return out
    if len(np.unique(y_true)) < 2:
        return out

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
        return out
    except Exception:
        out["auroc"] = _roc_auc_score_manual(y_true, y_prob)
        out["auprc"] = _average_precision_manual(y_true, y_prob)
        return out


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    y_pred = (y_prob >= float(thr)).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    tpr = safe_div(tp, tp + fn)       # recall / sensitivity
    rec = tpr
    tnr = safe_div(tn, tn + fp)       # specificity
    bal_acc = 0.5 * (tpr + tnr)

    prec = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * prec * rec, (prec + rec))
    jaccard = safe_div(tp, tp + fp + fn)

    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    eps = 1e-6
    p = np.clip(y_prob, eps, 1.0 - eps)
    logloss = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    brier = float(((y_prob - y_true) ** 2).mean())

    curve = binary_curve_metrics(y_true, y_prob)

    return {
        "thr": float(thr),
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "precision": float(prec),
        "prec": float(prec),
        "recall": float(rec),
        "rec": float(rec),
        "tpr": float(tpr),
        "sensitivity": float(tpr),
        "tnr": float(tnr),
        "specificity": float(tnr),
        "npv": float(npv),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "logloss": float(logloss),
        "brier": float(brier),
        "auroc": float(curve["auroc"]),
        "auprc": float(curve["auprc"]),
        "n": float(len(y_true)),
        "pos_n": float(int((y_true == 1).sum())),
        "neg_n": float(int((y_true == 0).sum())),
    }


def _thr_candidates(y_prob: np.ndarray, grid: int = 199, strategy: str = "linspace") -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    grid = max(10, int(grid))
    strategy = str(strategy).lower()

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, grid + 2)[1:-1]
        thr = np.quantile(y_prob, qs).astype(np.float64)
    else:
        thr = np.linspace(0.01, 0.99, grid).astype(np.float64)

    thr = np.clip(thr, 1e-4, 1.0 - 1e-4)
    thr = np.unique(thr)
    return thr


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "bal_acc",
    grid: int = 199,
    strategy: str = "linspace",
) -> Tuple[float, float]:
    metric = str(metric).lower()
    aliases = {
        "precision": "prec",
        "recall": "rec",
        "specificity": "tnr",
        "sensitivity": "tpr",
    }
    metric = aliases.get(metric, metric)

    best_thr = 0.5
    best_val = -1e18
    for thr in _thr_candidates(y_prob, grid=grid, strategy=strategy):
        m = compute_binary_metrics(y_true, y_prob, thr=float(thr))
        v = float(m.get(metric, m["bal_acc"]))
        if np.isnan(v):
            continue
        if v > best_val:
            best_val = v
            best_thr = float(thr)
    return float(best_thr), float(best_val)
