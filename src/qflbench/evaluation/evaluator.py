"""Prediction-level evaluator that freezes thresholds on validation data."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .metrics import multilabel_report, optimize_f1_thresholds


def evaluate_with_validation_thresholds(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    *,
    rare_labels: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> dict:
    """Fit thresholds on validation, then evaluate test exactly once."""

    thresholds, valid = optimize_f1_thresholds(y_validation, validation_probabilities)
    validation = multilabel_report(
        y_validation, validation_probabilities, thresholds=thresholds,
        rare_labels=rare_labels, label_names=label_names,
    )
    test = multilabel_report(
        y_test, test_probabilities, thresholds=thresholds,
        rare_labels=rare_labels, label_names=label_names,
    )
    return {
        "threshold_source": "validation_per_label_f1",
        "thresholds": thresholds.tolist(),
        "threshold_valid": valid.tolist(),
        "validation": validation,
        "test": test,
    }
