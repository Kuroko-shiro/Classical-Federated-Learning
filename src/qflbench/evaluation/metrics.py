"""Leakage-safe multilabel metrics for the canonical IU X-ray protocol.

Undefined per-label values are represented by NaN and excluded from macro
averages.  In particular, a label needs both a positive and a negative example
for AUROC/AUPRC and the canonical per-label F1 report.  The number of valid
labels is always reported so a deceptively high mean cannot hide label loss.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, fn


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn = _binary_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def define_rare_labels(
    y_train: np.ndarray, *, bottom_fraction: float = 0.25,
) -> list[int]:
    """Freeze rare labels from training prevalence only (never test results)."""

    y_train = np.asarray(y_train)
    if y_train.ndim != 2 or y_train.shape[1] == 0:
        raise ValueError("y_train must have shape [samples, labels]")
    if not 0 < bottom_fraction <= 1:
        raise ValueError("bottom_fraction must satisfy 0 < value <= 1")
    count = max(1, int(np.ceil(y_train.shape[1] * bottom_fraction)))
    positives = y_train.sum(axis=0)
    return sorted(np.argsort(positives, kind="stable")[:count].astype(int).tolist())


def optimize_f1_thresholds(
    y_val: np.ndarray,
    probabilities: np.ndarray,
    *,
    default: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one F1-maximising threshold per label using validation only.

    Ties are resolved toward the threshold closest to 0.5, then toward the
    larger threshold. Labels without both classes retain ``default`` and are
    marked invalid.
    """

    y_val = np.asarray(y_val, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_val.shape != probabilities.shape or y_val.ndim != 2:
        raise ValueError("y_val and probabilities must have equal [N, C] shape")
    thresholds = np.full(y_val.shape[1], float(default), dtype=np.float64)
    valid = np.zeros(y_val.shape[1], dtype=bool)
    for label in range(y_val.shape[1]):
        yt = y_val[:, label]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        valid[label] = True
        scores = probabilities[:, label]
        candidates = np.unique(np.concatenate(([0.0, default, 1.0], scores)))
        best = (-1.0, -np.inf, -np.inf, float(default))
        for threshold in candidates:
            _, _, f1 = _precision_recall_f1(yt, scores >= threshold)
            key = (f1, -abs(float(threshold) - default), float(threshold), float(threshold))
            if key > best:
                best = key
        thresholds[label] = best[-1]
    return thresholds, valid


def multilabel_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Optional[Sequence[float]] = None,
    rare_labels: Optional[Iterable[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> dict:
    """Return scalar summary plus transparent per-label metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != probabilities.shape or y_true.ndim != 2:
        raise ValueError("y_true and probabilities must have equal [N, C] shape")
    labels = y_true.shape[1]
    names = list(label_names or [str(index) for index in range(labels)])
    if len(names) != labels:
        raise ValueError("label_names length does not match y_true")
    selected_thresholds = np.asarray(
        thresholds if thresholds is not None else np.full(labels, 0.5),
        dtype=np.float64,
    )
    if selected_thresholds.shape != (labels,):
        raise ValueError("thresholds must have shape [C]")
    rare = set(int(index) for index in (rare_labels or []))
    predictions = probabilities >= selected_thresholds[None, :]
    fixed_predictions = probabilities >= 0.5

    per_label = []
    f1_values, fixed_f1_values, aurocs, auprcs = [], [], [], []
    all_tp = all_fp = all_fn = 0
    for label in range(labels):
        yt = y_true[:, label]
        positives = int(yt.sum())
        negatives = int(len(yt) - positives)
        evaluable = positives > 0 and negatives > 0
        if evaluable:
            precision, recall, f1 = _precision_recall_f1(yt, predictions[:, label])
            _, _, fixed_f1 = _precision_recall_f1(yt, fixed_predictions[:, label])
            auroc = float(roc_auc_score(yt, probabilities[:, label]))
            auprc = float(average_precision_score(yt, probabilities[:, label]))
            tp, fp, fn = _binary_counts(yt, predictions[:, label])
            all_tp += tp
            all_fp += fp
            all_fn += fn
        else:
            precision = recall = f1 = fixed_f1 = auroc = auprc = float("nan")
        f1_values.append(f1)
        fixed_f1_values.append(fixed_f1)
        aurocs.append(auroc)
        auprcs.append(auprc)
        per_label.append({
            "label_index": label,
            "label": names[label],
            "positive_count": positives,
            "negative_count": negatives,
            "prevalence": float(positives / len(yt)) if len(yt) else float("nan"),
            "threshold": float(selected_thresholds[label]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_threshold_0.5": fixed_f1,
            "auroc": auroc,
            "auprc": auprc,
            "evaluable": evaluable,
            "rare": label in rare,
        })

    flat_y = y_true.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    micro_auroc = (
        float(roc_auc_score(flat_y, flat_probabilities))
        if np.unique(flat_y).size == 2 else float("nan")
    )
    micro_auprc = (
        float(average_precision_score(flat_y, flat_probabilities))
        if np.unique(flat_y).size == 2 else float("nan")
    )
    _, _, micro_f1 = _precision_recall_f1(flat_y, predictions.reshape(-1))
    valid_f1 = np.asarray(f1_values, dtype=np.float64)
    finite_f1 = valid_f1[np.isfinite(valid_f1)]
    bottom_three = np.sort(finite_f1)[:3]
    rare_f1 = [f1_values[index] for index in sorted(rare) if 0 <= index < labels]
    summary = {
        "accuracy": float(np.mean(np.all(predictions == y_true, axis=1))) if len(y_true) else float("nan"),
        "hamming_acc": float(np.mean(predictions == y_true)) if y_true.size else float("nan"),
        "macro_f1": _safe_nanmean(f1_values),
        "macro_f1_val_optimized": _safe_nanmean(f1_values),
        "macro_f1_threshold_0.5": _safe_nanmean(fixed_f1_values),
        "micro_f1": float(micro_f1),
        "auroc": _safe_nanmean(aurocs),
        "auprc": _safe_nanmean(auprcs),
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "rare_label_macro_f1": _safe_nanmean(rare_f1),
        "bottom3_label_mean_f1": _safe_nanmean(bottom_three),
        "worst_label_f1": float(finite_f1.min()) if len(finite_f1) else float("nan"),
        "valid_f1_labels": int(np.isfinite(f1_values).sum()),
        "valid_auroc_labels": int(np.isfinite(aurocs).sum()),
        "valid_auprc_labels": int(np.isfinite(auprcs).sum()),
    }
    return {
        "summary": summary,
        "per_label": per_label,
        "thresholds": selected_thresholds.tolist(),
        "rare_label_indices": sorted(rare),
    }
