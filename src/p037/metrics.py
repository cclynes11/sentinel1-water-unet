"""Confusion counts and scores. The same definitions as ``analysis/usf_eval.py`` (zero-shot run)."""

from __future__ import annotations

import numpy as np

from .datasets.base import INVALID

NAN = float("nan")


def confusion(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, positive: list[int],
              exclude: list[int] | None = None) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) of a boolean prediction against the label values in ``positive``.

    Pixels that are invalid, unlabelled, or of a class in ``exclude`` are left out. One-vs-rest
    scoring of one flood class excludes the other flood class: the model has a single flood
    output, so a detected urban-flood pixel is not a false alarm for open flood."""
    m = valid & (gt != INVALID)
    if exclude:
        m &= ~np.isin(gt, exclude)
    pos = np.isin(gt, positive)
    tp = int(np.sum(m & pred & pos))
    fp = int(np.sum(m & pred & ~pos))
    fn = int(np.sum(m & ~pred & pos))
    tn = int(np.sum(m & ~pred & ~pos))
    return tp, fp, fn, tn


def scores(tp: float, fp: float, fn: float, tn: float = 0) -> dict[str, float]:
    """F1, IoU, precision, recall and accuracy; NaN where the denominator is zero."""
    total = tp + fp + fn + tn
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else NAN,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else NAN,
        "precision": tp / (tp + fp) if (tp + fp) else NAN,
        "recall": tp / (tp + fn) if (tp + fn) else NAN,
        "accuracy": (tp + tn) / total if total else NAN,
    }


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits."""
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def class_sets(classes: dict[int, str]) -> dict[str, tuple[list[int], list[int]]]:
    """Scoring sets: each class one-vs-rest (other classes excluded), and ``any`` (all classes
    positive, nothing excluded). ``any`` is what the model is trained on."""
    values = list(classes)
    out = {name: ([v], [o for o in values if o != v]) for v, name in classes.items()}
    out["any"] = (values, [])
    return out
