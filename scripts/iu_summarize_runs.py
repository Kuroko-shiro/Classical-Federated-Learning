"""Aggregate validation-selected test results across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/iu")
    ap.add_argument("--output", default="results/iu/summary.csv")
    return ap.parse_args()


def group_key(config):
    args = config["config"]
    return (
        args.get("mode", "fl"),
        args.get("scenario", "baseline"),
        args.get("method", args.get("mode", "unknown")),
        args.get("alpha"),
        args.get("mm_ratio", "full"),
        tuple(args.get("embed_dims") or ()),
    )


def main():
    args = parse_args()
    grouped = defaultdict(list)
    for test_path in sorted(Path(args.results_root).glob("*/test.json")):
        config_path = test_path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        grouped[group_key(config)].append({
            "seed": config["config"].get("seed", 0),
            "metrics": test["metrics"],
            "run_dir": str(test_path.parent),
        })

    rows = []
    for key, runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        mode, scenario, method, alpha, ratio, embed_dims = key
        seeds = [run["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"duplicate seed in group {key}: {seeds}")
        for metric in ("auroc", "macro_f1", "auprc"):
            values = np.asarray([run["metrics"][metric] for run in runs], dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append({
                "mode": mode, "scenario": scenario, "method": method,
                "alpha": alpha, "mm_ratio": ratio,
                "embed_dims": "-".join(str(value) for value in embed_dims),
                "metric": metric, "n": len(values), "seeds": ",".join(map(str, sorted(seeds))),
                "mean": mean, "std": std,
                "ci95_low": mean - half, "ci95_high": mean + half,
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "mode", "scenario", "method", "alpha", "mm_ratio", "embed_dims",
        "metric", "n", "seeds", "mean", "std", "ci95_low", "ci95_high",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} aggregate rows -> {output}")


if __name__ == "__main__":
    main()
