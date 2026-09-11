"""Evaluation metrics for ranking / regression multitask models."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _finite_pairs(y_true, y_pred)
    if len(yt) < 3:
        return float("nan")
    coef, _ = spearmanr(yt, yp)
    return float(coef)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _finite_pairs(y_true, y_pred)
    if len(yt) < 3:
        return float("nan")
    coef, _ = pearsonr(yt, yp)
    return float(coef)


def topk_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    true_top_frac: float = 0.10,
    pred_top_frac: float = 0.50,
) -> float:
    """Fraction of true top-X% molecules recovered in predicted top-Y%.

    Mirrors the proposal/COMET style top-10% / top-30% classification checks.
    """
    yt, yp = _finite_pairs(y_true, y_pred)
    n = len(yt)
    if n < 5:
        return float("nan")
    n_true = max(1, int(np.ceil(true_top_frac * n)))
    n_pred = max(1, int(np.ceil(pred_top_frac * n)))
    true_hits = set(np.argsort(-yt)[:n_true].tolist())
    pred_hits = set(np.argsort(-yp)[:n_pred].tolist())
    return float(len(true_hits & pred_hits) / n_true)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": spearman_corr(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "top10_in_top50": topk_hit_rate(y_true, y_pred, true_top_frac=0.10, pred_top_frac=0.50),
        "top30_in_top50": topk_hit_rate(y_true, y_pred, true_top_frac=0.30, pred_top_frac=0.50),
        "n": float(np.isfinite(y_true).sum()),
    }


def multitask_metrics(
    y_true: dict[str, np.ndarray],
    y_pred: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for task, yt in y_true.items():
        if task not in y_pred:
            continue
        out[task] = regression_metrics(yt, y_pred[task])
    return out
