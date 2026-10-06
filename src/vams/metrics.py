"""Threshold policy and reliability scores.

The decision threshold is chosen only from validation scores. Selection
maximizes recall of ACTION_NEEDED subject to a precision floor, because a
missed deadline costs more than an extra human review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

MIN_ACTION_PRECISION = 0.70


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    f2: float
    policy: str


def _binary_scores(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    precision = float(precision_score(y_true, pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    f2 = float(fbeta_score(y_true, pred, beta=2, pos_label=1, zero_division=0))
    return precision, recall, f2


def select_threshold(
    y_true,
    p_action,
    min_precision: float = MIN_ACTION_PRECISION,
) -> ThresholdChoice:
    """Pick an ACTION_NEEDED threshold without looking at the test set.

    Search a fixed grid. Prefer the highest recall whose precision is at least
    ``min_precision``. Ties break toward higher precision, then the threshold
    closest to 0.5. If nothing clears the floor, fall back to the best F2.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_action, dtype=float)
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)

    best_ok: tuple[tuple[float, float, float], ThresholdChoice] | None = None
    best_f2: tuple[tuple[float, float, float], ThresholdChoice] | None = None
    for threshold in grid:
        pred = (p >= threshold).astype(int)
        precision, recall, f2 = _binary_scores(y, pred)
        choice = ThresholdChoice(float(threshold), precision, recall, f2, "")
        f2_key = (f2, recall, -abs(float(threshold) - 0.5))
        if best_f2 is None or f2_key > best_f2[0]:
            best_f2 = (f2_key, choice)
        if precision + 1e-12 >= min_precision:
            ok_key = (recall, precision, -abs(float(threshold) - 0.5))
            if best_ok is None or ok_key > best_ok[0]:
                best_ok = (ok_key, choice)

    if best_ok is not None:
        chosen = best_ok[1]
        policy = f"max_recall_at_precision_{min_precision:.2f}"
    else:
        assert best_f2 is not None
        chosen = best_f2[1]
        policy = "max_f2_fallback"
    return ThresholdChoice(chosen.threshold, chosen.precision, chosen.recall, chosen.f2, policy)


def select_suspicious_threshold(y_true, scores) -> tuple[float, str, float, float]:
    """Choose a suspicious-score cutoff on validation F1, then recall."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if int(y.sum()) == 0:
        return 2.0, "default_no_positive_validation_rows", 0.0, 0.0
    candidates = sorted(set(np.round(s, 2).tolist()) | {0.0, 1.0, 1.5, 2.0, 3.0, 4.0})
    best_key = None
    best = (2.0, 0.0, 0.0)
    for threshold in candidates:
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        key = (f1, recall, precision, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), precision, recall)
    return best[0], "max_f1_then_recall", best[1], best[2]


def expected_calibration_error(y_true, p_action, n_bins: int = 10) -> float:
    """Equal-width ECE for the positive class."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_action, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)
