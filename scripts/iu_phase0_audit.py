#!/usr/bin/env python3
"""Create P0-A1/A2 environment, dataset and split audit artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qflbench.data.iu_xray_labels import CHEXPERT_LABELS
from qflbench.data.iu_xray_prep import build_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--split", default="splits/iu_split.json")
    parser.add_argument("--img-cache", default=None)
    parser.add_argument("--output", default="results/phase0/data_audit")
    parser.add_argument("--environment-output", default="environment")
    return parser.parse_args()


def sha256_path(path: str) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    files = [target] if target.is_file() else sorted(item for item in target.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(target) if target.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"unavailable: {error}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def main():
    args = parse_args()
    output = Path(args.output)
    environment = Path(args.environment_output)
    output.mkdir(parents=True, exist_ok=True)
    environment.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        args.reports, args.projections, args.images,
        require_findings=True, require_frontal=False,
    )
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if len(manifest) != 3337:
        raise RuntimeError(f"canonical manifest must contain 3337 studies, got {len(manifest)}")
    groups = {
        "train_pool": set(map(int, split["train_pool"])),
        "public": set(map(int, split["public"])),
        "test": set(map(int, split["test"])),
    }
    overlaps = {
        f"{left}__{right}": len(groups[left] & groups[right])
        for index, left in enumerate(groups)
        for right in list(groups)[index + 1:]
    }
    integrity = {
        "manifest_size": len(manifest),
        "train_pool_size": len(groups["train_pool"]),
        "public_size": len(groups["public"]),
        "test_size": len(groups["test"]),
        "overlaps": overlaps,
        "expected": {"manifest": 3337, "train_pool": 2510, "public": 200, "test": 627},
        "valid": (
            len(groups["train_pool"]) == 2510
            and len(groups["public"]) == 200
            and len(groups["test"]) == 627
            and not any(overlaps.values())
        ),
    }
    partition_checks = {}
    for alpha, partitions in split["by_alpha"].items():
        assigned = [int(index) for indices in partitions.values() for index in indices]
        partition_checks[alpha] = {
            "samples": len(assigned),
            "unique_samples": len(set(assigned)),
            "covers_train_pool": set(assigned) == groups["train_pool"],
            "disjoint": len(assigned) == len(set(assigned)),
        }
    integrity["dirichlet_partitions"] = partition_checks
    integrity["valid"] = integrity["valid"] and all(
        check["covers_train_pool"] and check["disjoint"]
        for check in partition_checks.values()
    )
    if not integrity["valid"]:
        raise RuntimeError(f"split integrity failed: {integrity}")

    hashes = {
        "reports": sha256_path(args.reports),
        "projections": sha256_path(args.projections),
        "images": sha256_path(args.images),
        "split": sha256_path(args.split),
    }
    if args.img_cache:
        hashes["img_cache"] = sha256_path(args.img_cache)
    (output / "dataset_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (output / "split_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    count_rows, prevalence_rows, partition_rows = [], [], []
    for alpha, partitions in sorted(split["by_alpha"].items(), key=lambda item: float(item[0])):
        for client_id, indices in sorted(partitions.items(), key=lambda item: int(item[0])):
            labels = np.stack([manifest[int(index)]["label"] for index in indices])
            count_rows.append({"alpha": alpha, "client_id": client_id, "samples": len(indices)})
            for label_index, label in enumerate(CHEXPERT_LABELS):
                positives = int(labels[:, label_index].sum())
                prevalence_rows.append({
                    "alpha": alpha, "client_id": client_id,
                    "label_index": label_index, "label": label,
                    "positive_count": positives,
                    "prevalence": positives / len(indices) if indices else float("nan"),
                })
        sizes = np.asarray([len(indices) for indices in partitions.values()], dtype=float)
        client_prevalence = []
        for indices in partitions.values():
            labels = np.stack([manifest[int(index)]["label"] for index in indices])
            client_prevalence.append(labels.mean(axis=0))
        client_prevalence = np.stack(client_prevalence)
        mean_prevalence = client_prevalence.mean(axis=0)
        for label_index, label in enumerate(CHEXPERT_LABELS):
            probabilities = client_prevalence[:, label_index]
            jsd = binary_entropy(float(mean_prevalence[label_index])) - float(
                np.mean([binary_entropy(float(value)) for value in probabilities])
            )
            partition_rows.append({
                "alpha": alpha,
                "label_index": label_index,
                "label": label,
                "client_size_cv": float(sizes.std() / sizes.mean()),
                "label_js_divergence": jsd,
                "clients_without_positive": int((probabilities == 0).sum()),
                "minimum_client_prevalence": float(probabilities.min()),
                "maximum_client_prevalence": float(probabilities.max()),
            })
    write_csv(output / "client_counts.csv", count_rows)
    write_csv(output / "label_prevalence.csv", prevalence_rows)
    noniid_output = output.parent / "noniid_audit"
    noniid_output.mkdir(parents=True, exist_ok=True)
    write_csv(noniid_output / "partition_statistics.csv", partition_rows)
    (output / "data_audit.md").write_text(
        "# Phase 0 data audit\n\n"
        f"- manifest: {len(manifest)}\n- train/public/test: 2510 / 200 / 627\n"
        f"- overlap checks: {overlaps}\n- status: PASS\n",
        encoding="utf-8",
    )

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_hash": command_output(["git", "rev-parse", "HEAD"]),
        "torch": command_output([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "transformers": command_output([sys.executable, "-c", "import transformers; print(transformers.__version__)"]),
        "accelerator": command_output([
            sys.executable, "-c",
            "import torch; print({'mps': torch.backends.mps.is_available(), 'cuda': torch.cuda.is_available()})",
        ]),
    }
    (environment / "environment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    (environment / "requirements-lock.txt").write_text(
        command_output([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8",
    )
    (environment / "conda-history.yml").write_text(
        command_output(["conda", "env", "export", "--from-history"]) + "\n", encoding="utf-8",
    )
    print(f"Phase 0 audit: PASS -> {output}")


if __name__ == "__main__":
    main()
