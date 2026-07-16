"""Dependency-light tests for the Phase-0 IU X-ray protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from qflbench.data.iu_xray_labels import CHEXPERT_LABELS, active_labels, encode_problems
from qflbench.data.iu_xray_prep import build_manifest, partition_clients
from qflbench.experiments.iu_protocol import (
    BestCheckpoint,
    CommunicationLedger,
    load_checkpoint,
    load_iu_split,
)


class LabelTests(unittest.TestCase):
    def test_positive_labels_override_no_finding(self):
        labels = active_labels("normal; cardiomegaly; pleural effusion; PICC line")
        self.assertEqual(labels, ["Cardiomegaly", "Pleural Effusion", "Support Devices"])

    def test_unknown_is_not_silently_normal(self):
        self.assertEqual(float(encode_problems("unmapped-term").sum()), 0.0)
        self.assertEqual(len(CHEXPERT_LABELS), 14)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_uid_sorted_and_uses_all_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([
                {"uid": 2, "Problems": "normal", "findings": "clear"},
                {"uid": 1, "Problems": "pneumonia", "findings": "opacity"},
                {"uid": 3, "Problems": "normal", "findings": None},
            ]).to_csv(root / "reports.csv", index=False)
            pd.DataFrame([
                {"uid": 1, "filename": "b.png", "projection": "Lateral"},
                {"uid": 2, "filename": "c.png", "projection": "Frontal"},
                {"uid": 1, "filename": "a.png", "projection": "Frontal"},
            ]).to_csv(root / "projections.csv", index=False)
            manifest = build_manifest(
                str(root / "reports.csv"), str(root / "projections.csv"), str(root / "images")
            )
            self.assertEqual([item["uid"] for item in manifest], ["1", "2"])
            self.assertEqual(manifest[0]["filenames"], ["a.png", "b.png"])

    def test_dirichlet_partition_is_complete_and_deterministic(self):
        manifest = [
            {"label": np.eye(14, dtype=np.float32)[i % 4]} for i in range(80)
        ]
        first = partition_clients(manifest, range(80), num_clients=4, alpha=0.3, seed=7)
        second = partition_clients(manifest, range(80), num_clients=4, alpha=0.3, seed=7)
        self.assertEqual(first, second)
        assigned = [i for values in first.values() for i in values]
        self.assertEqual(sorted(assigned), list(range(80)))
        self.assertEqual(len(assigned), len(set(assigned)))


class ProtocolTests(unittest.TestCase):
    def _split_file(self, root: Path) -> Path:
        raw = {
            "meta": {"manifest_size": 30, "clients": 2},
            "public": [0, 1],
            "test": [2, 3, 4, 5],
            "train_pool": list(range(6, 30)),
            "by_alpha": {"1.0": {"0": list(range(6, 18)), "1": list(range(18, 30))}},
        }
        path = root / "split.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_validation_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = load_iu_split(
                str(self._split_file(Path(tmp))), manifest_size=30, alpha=1.0,
                clients=2, train_subset=24, test_subset=None, val_fraction=0.25, val_seed=9,
            )
            train = {i for values in split.train_by_client.values() for i in values}
            val = set(split.validation)
            self.assertFalse(train & val)
            self.assertFalse((train | val) & set(split.test))
            self.assertEqual(len(train) + len(val), 24)
            self.assertEqual(len(split.test), 4)

    def test_client_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_iu_split(
                    str(self._split_file(Path(tmp))), manifest_size=30, alpha=1.0, clients=1
                )

    def test_partition_coverage_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._split_file(Path(tmp))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["by_alpha"]["1.0"]["1"].remove(29)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_iu_split(
                    str(path), manifest_size=30, alpha=1.0, clients=2,
                    train_subset=24,
                )

    def test_communication_ledger_splits_direction_and_client(self):
        ledger = CommunicationLedger()
        ledger.start_round(0)
        ledger.record(0, "upload", np.zeros((2, 3), dtype=np.float32))
        ledger.record(1, "upload", np.zeros(4, dtype=np.int8))
        ledger.record(0, "download", np.zeros(2, dtype=np.float64))
        row = ledger.finish_round()
        self.assertEqual(row["upload_bytes"], 28)
        self.assertEqual(row["download_bytes"], 16)
        self.assertEqual(row["clients"]["1"]["upload_bytes"], 4)
        self.assertGreater(row["serialized_total_bytes"], row["total_bytes"])
        self.assertEqual(
            row["qkd_otp"]["required_key_bits"],
            8 * row["serialized_total_bytes"],
        )

    def test_best_checkpoint_uses_validation_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.npz"
            tracker = BestCheckpoint(str(path))
            self.assertTrue(tracker.update(
                0, {"auroc": 0.6}, {"global": {"w": np.array([1], dtype=np.float32)}},
                metadata={"method": "fedavg"},
            ))
            self.assertFalse(tracker.update(
                1, {"auroc": 0.5}, {"global": {"w": np.array([2], dtype=np.float32)}}
            ))
            self.assertTrue(tracker.update(
                2, {"auroc": 0.7}, {"global": {"w": np.array([3], dtype=np.float32)}}
            ))
            metadata, models, arrays = load_checkpoint(str(path))
            self.assertEqual(metadata["best_round"], 2)
            self.assertEqual(models["global"]["w"].tolist(), [3.0])
            self.assertEqual(arrays, {})


if __name__ == "__main__":
    unittest.main()
