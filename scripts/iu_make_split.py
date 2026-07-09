"""Freeze the data split ONCE into a JSON file, shared by ALL scenarios/methods.

This is the root-cause fix for the repeated re-runs. Until now the split was
recomputed inside every runner, so any code change / arg difference / run-order
quietly shifted client assignments and silently broke cross-scenario comparison.

From now on: run this ONCE to produce splits/iu_split.json, then every runner
LOADS that file instead of recomputing. The split can never drift again.

The file contains, for each alpha:
  - public      : 200 indices (held out from train/test, used by FedMD/LOOT)
  - test        : test indices (same across all methods)
  - train_pool  : the full training pool (same across all methods)
  - clients     : {client_id: [indices]} Dirichlet partition for THIS alpha
Indices refer to positions in the manifest built with the FIXED recipe below
(require_findings=True, require_frontal=False) — that recipe is also frozen here.

Run once:
    python scripts/iu_make_split.py \
        --reports data/indiana_reports.csv \
        --projections data/indiana_projections.csv \
        --alphas 0.1 1.0 \
        --clients 4 --out splits/iu_split.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from qflbench.data.iu_xray_prep import build_manifest, partition_clients


# FROZEN manifest recipe — must match what every runner uses.
MANIFEST_RECIPE = dict(require_findings=True, require_frontal=False)
PUBLIC_SIZE = 200
TEST_FRAC = 0.2
SPLIT_SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", default="data/images/images_normalized",
                    help="only used to build manifest paths; not read here")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--out", default="splits/iu_split.json")
    args = ap.parse_args()

    manifest = build_manifest(args.reports, args.projections, args.images,
                              **MANIFEST_RECIPE)
    n = len(manifest)
    print(f"manifest: {n} samples (recipe={MANIFEST_RECIPE})")

    # fixed public/test/train_pool split (case-3 policy)
    rng = np.random.default_rng(SPLIT_SEED)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    public = sorted(all_idx[:PUBLIC_SIZE].tolist())
    rest = all_idx[PUBLIC_SIZE:]
    n_test = int(round(len(rest) * TEST_FRAC))
    test = sorted(rest[:n_test].tolist())
    train_pool = sorted(rest[n_test:].tolist())
    print(f"public={len(public)}  test={len(test)}  train_pool={len(train_pool)}")

    out = {
        "meta": {
            "manifest_size": n, "recipe": MANIFEST_RECIPE,
            "public_size": PUBLIC_SIZE, "test_frac": TEST_FRAC,
            "split_seed": SPLIT_SEED, "clients": args.clients,
        },
        "public": public,
        "test": test,
        "train_pool": train_pool,
        "by_alpha": {},
    }

    # per-alpha client partitions of the SAME train_pool
    for a in args.alphas:
        part = partition_clients(manifest, train_pool, num_clients=args.clients,
                                 scheme="dirichlet", alpha=a, seed=SPLIT_SEED)
        sizes = {c: len(v) for c, v in part.items()}
        out["by_alpha"][str(a)] = {str(c): v for c, v in part.items()}
        print(f"  alpha={a}: client sizes {sizes}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nsaved frozen split -> {args.out}")
    print("ALL runners must now load this file. Split will never drift again.")


if __name__ == "__main__":
    main()
