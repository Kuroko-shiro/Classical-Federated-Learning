#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qflbench.mimic import audit_access, audit_cohort, build_fl_partition, evaluate_complementarity_gate


def main() -> None:
    p = argparse.ArgumentParser(description="MIMIC-IV + CXR benchmark validation harness")
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("audit-access"); a.add_argument("--mimic-iv-root"); a.add_argument("--mimic-cxr-root"); a.add_argument("--benchmark-repo"); a.add_argument("--output-dir", required=True)
    c = sp.add_parser("audit-cohort"); c.add_argument("--manifest", required=True); c.add_argument("--output-dir", required=True)
    g = sp.add_parser("gate0"); g.add_argument("--metrics", required=True); g.add_argument("--output-dir", required=True); g.add_argument("--leakage-pass", choices=["true", "false"], default="true"); g.add_argument("--min-seeds", type=int, default=3)
    f = sp.add_parser("partition"); f.add_argument("--manifest", required=True); f.add_argument("--output-dir", required=True); f.add_argument("--k", type=int, default=10); f.add_argument("--seed", type=int, default=0); f.add_argument("--alpha", default="iid"); f.add_argument("--complete-ratio", type=float, default=.5); f.add_argument("--min-positive", type=int, default=1); f.add_argument("--min-negative", type=int, default=1)
    args = p.parse_args()
    if args.cmd == "audit-access": result = audit_access(args.mimic_iv_root, args.mimic_cxr_root, args.output_dir, args.benchmark_repo)
    elif args.cmd == "audit-cohort": result = audit_cohort(args.manifest, args.output_dir)
    elif args.cmd == "gate0": result = evaluate_complementarity_gate(args.metrics, args.output_dir, leakage_pass=args.leakage_pass == "true", min_seeds_for_go=args.min_seeds)
    else:
        alpha = None if args.alpha.lower() in {"iid", "inf", "infinity"} else float(args.alpha)
        result = build_fl_partition(args.manifest, args.output_dir, k=args.k, seed=args.seed, alpha=alpha, complete_ratio=args.complete_ratio, min_positive=args.min_positive, min_negative=args.min_negative)["summary"]
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__": main()
