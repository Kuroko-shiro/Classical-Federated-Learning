#!/usr/bin/env python3
"""Build the Phase-0 canonical/legacy/invalid experiment registry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "run_id", "scenario", "method", "alpha", "split_seed", "train_seed",
    "embed_dims", "modality_ratio", "train_samples", "public_samples",
    "test_samples", "rounds", "local_epochs", "learning_rate", "git_hash",
    "checkpoint_available", "communication_valid", "evaluation_type", "status", "notes",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/iu")
    parser.add_argument("--output", default="results/phase0")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, canonical, legacy, invalid = [], [], [], []
    for config_path in sorted(Path(args.results_root).glob("*/config.json")):
        run = config_path.parent
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        protocol = data.get("protocol", {})
        test_path = run / "test.json"
        communication = run / "communication.jsonl"
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
        canonical_run = (
            protocol.get("test_size") == 627
            and test.get("evaluation_policy") == "test evaluated once after validation selection"
            and (run / "best_validation.npz").exists()
        )
        status = "canonical candidate" if canonical_run else ("legacy" if test_path.exists() else "invalid")
        row = {
            "run_id": run.name,
            "scenario": config.get("scenario", "baseline"),
            "method": config.get("method", config.get("mode", "unknown")),
            "alpha": config.get("alpha"), "split_seed": 0,
            "train_seed": config.get("seed", 0),
            "embed_dims": config.get("embed_dims", config.get("embed_dim")),
            "modality_ratio": config.get("mm_ratio", "full"),
            "train_samples": protocol.get("train_size"),
            "public_samples": protocol.get("public_size"),
            "test_samples": protocol.get("test_size"),
            "rounds": config.get("rounds", config.get("epochs")),
            "local_epochs": config.get("local_epochs"),
            "learning_rate": config.get("lr", config.get("effective_lr")),
            "git_hash": data.get("git_commit"),
            "checkpoint_available": (run / "best_validation.npz").exists(),
            "communication_valid": communication.exists() and communication.stat().st_size > 0,
            "evaluation_type": "canonical" if canonical_run else "legacy_recomputed",
            "status": status,
            "notes": "",
        }
        rows.append(row)
        {"canonical candidate": canonical, "legacy": legacy, "invalid": invalid}[status].append(run.name)
    with (output / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for filename, values in (
        ("canonical_runs.yaml", canonical), ("legacy_runs.yaml", legacy),
    ):
        (output / filename).write_text("runs:\n" + "".join(f"  - {value}\n" for value in values), encoding="utf-8")
    (output / "excluded_runs.md").write_text(
        "# Excluded runs\n\n" + "".join(f"- {value}\n" for value in invalid), encoding="utf-8",
    )
    print(f"wrote {len(rows)} runs -> {output / 'experiment_registry.csv'}")


if __name__ == "__main__":
    main()
