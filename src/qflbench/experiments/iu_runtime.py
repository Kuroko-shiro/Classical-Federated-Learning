"""Small runtime helpers for the Phase-0 IU X-ray command-line runners."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


METRIC_FIELDS = (
    "accuracy", "hamming_acc",
    "macro_f1", "macro_f1_val_optimized", "macro_f1_threshold_0.5",
    "micro_f1", "auroc", "auprc", "micro_auroc", "micro_auprc",
    "rare_label_macro_f1", "bottom3_label_mean_f1", "worst_label_f1",
    "valid_f1_labels", "valid_auroc_labels", "valid_auprc_labels",
)

COMMUNICATION_FIELDS = (
    "upload_bytes", "download_bytes", "total_bytes",
    "cumulative_upload_bytes", "cumulative_download_bytes",
    "cumulative_total_bytes", "serialized_upload_bytes",
    "serialized_download_bytes", "serialized_total_bytes",
    "cumulative_serialized_upload_bytes",
    "cumulative_serialized_download_bytes",
    "cumulative_serialized_total_bytes",
)


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict:
    if not rows:
        raise ValueError("cannot average an empty metric list")
    keys = [key for key in METRIC_FIELDS if any(key in row for row in rows)]
    output = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        output[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return output


def rare_labels_from_manifest(manifest, train_indices, *, bottom_fraction: float = 0.25):
    from ..evaluation.metrics import define_rare_labels

    labels = np.stack([np.asarray(manifest[index]["label"]) for index in train_indices])
    return define_rare_labels(labels, bottom_fraction=bottom_fraction)


def canonical_backend_evaluation(backend, validation_loader, test_loader, *, rare_labels):
    """Validation threshold fitting followed by exactly one test prediction pass."""

    from ..data.iu_xray_labels import CHEXPERT_LABELS
    from ..evaluation.evaluator import evaluate_with_validation_thresholds

    y_validation, validation_probabilities = backend.prediction_arrays(validation_loader)
    y_test, test_probabilities = backend.prediction_arrays(test_loader)
    return evaluate_with_validation_thresholds(
        y_validation, validation_probabilities, y_test, test_probabilities,
        rare_labels=rare_labels, label_names=CHEXPERT_LABELS,
    )


def canonical_client_evaluation(client_rows, *, rare_labels):
    """Evaluate heterogeneous clients in their own modality/width configuration."""

    reports = []
    for client_id, backend, validation_loader, test_loader in client_rows:
        report = canonical_backend_evaluation(
            backend, validation_loader, test_loader, rare_labels=rare_labels,
        )
        reports.append({"client_id": int(client_id), **report})
    metrics = mean_metrics([report["test"]["summary"] for report in reports])
    return metrics, {"per_client": reports, "rare_label_indices": list(rare_labels)}


def git_revision(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "nogit"


class RunArtifacts:
    """Write a self-contained run directory without test-trajectory leakage."""

    def __init__(
        self,
        *,
        root: str,
        run_name: str,
        config: Mapping[str, object],
        protocol: Mapping[str, object],
        repo_dir: str,
        extra_validation_fields: Sequence[str] = (),
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.validation_path = self.run_dir / "validation.csv"
        self.communication_path = self.run_dir / "communication.jsonl"
        self.checkpoint_path = self.run_dir / "best_validation.npz"
        self.test_path = self.run_dir / "test.json"
        full_config = {
            "config": dict(config),
            "protocol": dict(protocol),
            "git_commit": git_revision(repo_dir),
            "created_at": datetime.now().astimezone().isoformat(),
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(full_config, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._extra_validation_fields = tuple(extra_validation_fields)
        fields = [
            "round", *METRIC_FIELDS, "seconds",
            *COMMUNICATION_FIELDS,
            *self._extra_validation_fields,
        ]
        self._validation_file = self.validation_path.open("w", newline="", encoding="utf-8")
        self._validation_writer = csv.DictWriter(self._validation_file, fieldnames=fields)
        self._validation_writer.writeheader()
        self._communication_file = self.communication_path.open("w", encoding="utf-8")

    def log_validation(
        self, round_index: int, metrics: Mapping[str, float], seconds: float,
        communication: Mapping[str, object], *, extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        row = {field: None for field in METRIC_FIELDS}
        row.update({field: metrics.get(field) for field in METRIC_FIELDS})
        row.update({
            "round": int(round_index),
            "seconds": round(float(seconds), 3),
            **{key: communication.get(key, 0) for key in COMMUNICATION_FIELDS},
        })
        extra = dict(extra or {})
        row.update({field: extra.get(field) for field in self._extra_validation_fields})
        self._validation_writer.writerow(row)
        self._validation_file.flush()
        self.log_communication(communication)

    def log_communication(self, communication: Mapping[str, object]) -> None:
        """Record a communication-only phase such as MIN pre-training."""
        self._communication_file.write(json.dumps(communication, sort_keys=True) + "\n")
        self._communication_file.flush()

    def write_test(
        self, *, metrics: Mapping[str, float],
        checkpoint_metadata: Mapping[str, object],
        details: Optional[Mapping[str, object]] = None,
    ) -> None:
        payload = {
            "metrics": {key: float(value) for key, value in metrics.items()},
            "selected_checkpoint": dict(checkpoint_metadata),
            "evaluation_policy": "test evaluated once after validation selection",
        }
        if details is not None:
            payload["details"] = dict(details)
        self.test_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_json(self, relative_path: str, payload: object) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def append_jsonl(self, relative_path: str, payload: object) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return path

    def close(self) -> None:
        self._validation_file.close()
        self._communication_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
