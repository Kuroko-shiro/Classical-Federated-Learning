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


METRIC_FIELDS = ("accuracy", "hamming_acc", "macro_f1", "auroc", "auprc")


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict:
    if not rows:
        raise ValueError("cannot average an empty metric list")
    keys = [key for key in METRIC_FIELDS if key in rows[0]]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


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
            "upload_bytes", "download_bytes", "total_bytes",
            "cumulative_upload_bytes", "cumulative_download_bytes",
            "cumulative_total_bytes",
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
            **{key: communication[key] for key in (
                "upload_bytes", "download_bytes", "total_bytes",
                "cumulative_upload_bytes", "cumulative_download_bytes",
                "cumulative_total_bytes",
            )},
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

    def write_test(self, *, metrics: Mapping[str, float], checkpoint_metadata: Mapping[str, object]) -> None:
        payload = {
            "metrics": {key: float(value) for key, value in metrics.items()},
            "selected_checkpoint": dict(checkpoint_metadata),
            "evaluation_policy": "test evaluated once after validation selection",
        }
        self.test_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def close(self) -> None:
        self._validation_file.close()
        self._communication_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
