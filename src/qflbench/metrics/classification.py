"""Classification metrics tuned for imbalanced medical data.

We report accuracy plus macro-F1 and (for binary/probabilistic settings) AUROC and
AUPRC, because accuracy alone is misleading under class imbalance — the standard
caution in medical ML. macro-F1 weights classes equally; AUPRC is informative when
the positive class is rare.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(
    y_true: np.ndarray, logits: np.ndarray, num_classes: int
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    y_pred = probs.argmax(axis=1)

    out: Dict[str, float] = {}
    out["accuracy"] = float((y_pred == y_true).mean())

    try:
        from sklearn.metrics import f1_score
        out["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
    except Exception:
        out["macro_f1"] = _macro_f1_numpy(y_true, y_pred, num_classes)

    # AUROC / AUPRC: well-defined when >=2 classes present in y_true
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        present = np.unique(y_true)
        if num_classes == 2 and len(present) == 2:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
            out["auprc"] = float(average_precision_score(y_true, probs[:, 1]))
        elif len(present) > 1:
            # one-vs-rest macro for multiclass
            yb = np.eye(num_classes)[y_true]
            # restrict to present columns to avoid undefined AUC on absent classes
            cols = present
            out["auroc"] = float(
                roc_auc_score(yb[:, cols], probs[:, cols], average="macro", multi_class="ovr")
            )
            out["auprc"] = float(
                average_precision_score(yb[:, cols], probs[:, cols], average="macro")
            )
    except Exception:
        pass

    return out


def _macro_f1_numpy(y_true, y_pred, num_classes) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def aggregate_client_metrics(
    per_client: Dict[int, Dict[str, float]]
) -> Dict[str, float]:
    """Macro-average a metric dict across clients, plus worst-client and spread —
    the fairness view (per-client variance, worst client) from the design doc."""
    if not per_client:
        return {}
    keys = set().union(*[set(d.keys()) for d in per_client.values()])
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array(
            [d[k] for d in per_client.values() if k in d and not np.isnan(d[k])],
            dtype=np.float64,
        )
        if len(vals) == 0:
            continue
        out[f"mean_{k}"] = float(vals.mean())
        out[f"worst_{k}"] = float(vals.min())
        out[f"std_{k}"] = float(vals.std())
    return out


def multilabel_metrics(y_true: np.ndarray, probs: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
    """Metrics for MULTI-LABEL classification (IU X-ray, CheXpert 14 classes).

    y_true: [N, C] multi-hot ground truth.
    probs : [N, C] per-class probabilities (sigmoid outputs).
    Returns:
      accuracy : exact-match subset accuracy (all 14 labels correct) — strict.
                 Also report 'hamming_acc' (per-label correctness), which is the
                 more informative number for multi-label.
      macro_f1 : unweighted mean F1 over classes that appear in y_true.
      auroc/auprc : macro-averaged over classes with both positives & negatives.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if y_true.size == 0:
        return {"accuracy": 0.0, "hamming_acc": 0.0, "macro_f1": 0.0,
                "auroc": 0.0, "auprc": 0.0}
    preds = (probs >= threshold).astype(int)

    exact = float(np.mean(np.all(preds == y_true, axis=1)))
    hamming = float(np.mean(preds == y_true))

    # macro-F1 over classes present in y_true
    f1s = []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], preds[:, c]
        if yt.sum() == 0:
            continue
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # macro AUROC / AUPRC over evaluable classes
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        aurocs, auprcs = [], []
        for c in range(y_true.shape[1]):
            yt = y_true[:, c]
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            aurocs.append(roc_auc_score(yt, probs[:, c]))
            auprcs.append(average_precision_score(yt, probs[:, c]))
        auroc = float(np.mean(aurocs)) if aurocs else 0.0
        auprc = float(np.mean(auprcs)) if auprcs else 0.0
    except Exception:
        auroc = auprc = 0.0

    return {"accuracy": exact, "hamming_acc": hamming, "macro_f1": macro_f1,
            "auroc": auroc, "auprc": auprc}
