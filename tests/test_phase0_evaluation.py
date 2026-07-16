"""Dependency-light tests for revised Phase-0 metrics and diagnostics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from qflbench.communication.qkd_accounting import qkd_key_budget
from qflbench.communication.serializer import serialized_payload_nbytes
from qflbench.diagnostics.drift import update_drift_report
from qflbench.evaluation.metrics import (
    define_rare_labels,
    multilabel_report,
    optimize_f1_thresholds,
)


class MetricTests(unittest.TestCase):
    def test_validation_threshold_is_fitted_without_test_access(self):
        y = np.array([[0], [0], [1], [1]])
        probabilities = np.array([[0.1], [0.2], [0.3], [0.4]])
        thresholds, valid = optimize_f1_thresholds(y, probabilities)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(thresholds[0], 0.3)

    def test_undefined_label_is_nan_and_excluded(self):
        y = np.array([[0, 0], [1, 0], [1, 0]])
        probabilities = np.array([[0.1, 0.2], [0.8, 0.3], [0.9, 0.4]])
        report = multilabel_report(y, probabilities)
        self.assertEqual(report["summary"]["valid_auroc_labels"], 1)
        self.assertTrue(math.isnan(report["per_label"][1]["auroc"]))
        self.assertAlmostEqual(report["summary"]["auroc"], 1.0)

    def test_rare_labels_use_training_counts(self):
        y = np.zeros((10, 4), dtype=int)
        y[:8, 0] = 1
        y[:5, 1] = 1
        y[:2, 2] = 1
        y[:1, 3] = 1
        self.assertEqual(define_rare_labels(y, bottom_fraction=0.25), [3])


class CommunicationTests(unittest.TestCase):
    def test_serialized_bytes_include_metadata(self):
        payload = {"weight": np.zeros(4, dtype=np.float32)}
        self.assertGreater(serialized_payload_nbytes(payload), payload["weight"].nbytes)

    def test_qkd_otp_budget(self):
        budget = qkd_key_budget(1250, key_rates_bps=[10_000])
        self.assertEqual(budget["required_key_bits"], 10_000)
        self.assertEqual(budget["key_generation_seconds"]["10000"], 1.0)


class DriftTests(unittest.TestCase):
    def test_opposite_updates_have_zero_cancellation(self):
        reference = {"w": np.zeros(2)}
        report = update_drift_report(
            [{"w": np.ones(2)}, {"w": -np.ones(2)}],
            [reference, reference], [1, 1],
        )
        self.assertAlmostEqual(report["aggregation_cancellation_ratio"], 0.0)
        self.assertAlmostEqual(report["pairwise_update_cosines"][0]["cosine"], -1.0)


if __name__ == "__main__":
    unittest.main()
