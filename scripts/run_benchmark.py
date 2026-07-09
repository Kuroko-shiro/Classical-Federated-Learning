"""Benchmark runner — per-scenario, condition-driven, non-overwriting, with plots.

Key properties (per user's requirements):
  - Choose a scenario with --scenario {1,2} (3,4 added later).
  - Vary conditions on the command line (--alpha, --rounds, --target, --seeds);
    no need to edit code between runs.
  - Results are GROUPED BY SCENARIO and NEVER overwritten:
        results/scenarioN/runs/<method>_alpha<a>_<timestamp>/   (raw history)
        results/scenarioN/scenarioN_summary.csv                  (accumulated table)
        results/scenarioN/plots/*.png                            (auto visualization)
        results/all_runs_summary.csv                             (cross-scenario, for tax)
  - Each run also prints the mean±std comparison table to screen.

Examples:
    python scripts/run_benchmark.py --scenario 2 --rounds 40 --seeds 3
    python scripts/run_benchmark.py --scenario 1 --alpha 0.1 --rounds 40 --seeds 3
    python scripts/run_benchmark.py --scenario 2 --alpha 0.5            # sweep piece
    python scripts/run_benchmark.py --plot-only --scenario 2           # reg(re)draw plots
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qflbench.engine.simulator import run_experiment            # noqa: E402
from qflbench.metrics.analysis import summarize_run             # noqa: E402
from qflbench.metrics import results_store as store             # noqa: E402


# --------------------------------------------------------------------------- #
# Experiment configuration (now parameterized by alpha / rounds)
# --------------------------------------------------------------------------- #
def base_cfg(seed: int, alpha: float, rounds: int) -> Dict[str, Any]:
    return {
        "seed": seed,
        "num_clients": 6,
        "rounds": rounds,
        "client_fraction": 1.0,
        "dataset": {"name": "synthetic", "num_classes": 6, "n_per_class": 300,
                    "image_dim": 40, "text_dim": 30,
                    "image_signal": 0.45, "text_signal": 0.45, "noise": 2.5,
                    "subclusters": 5, "cluster_spread": 2.2, "seed": seed},
        "partition": {"name": "dirichlet", "alpha": alpha},
        "channel": {"name": "classical"},
        "local": {"local_epochs": 2, "lr": 0.05},
    }


def make_configs(scenario: int, seed: int, alpha: float, rounds: int,
                 run_dirs: Dict[str, str],
                 num_multimodal: int = 2) -> Dict[str, Dict[str, Any]]:
    """(method_name -> cfg) for the requested scenario, one seed."""
    cfgs: Dict[str, Dict[str, Any]] = {}

    if scenario == 1:
        # same model, same modality
        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": "full", "model_hetero": False,
                  "model": {"name": "mock", "embed_choices": [16], "share_encoders": True},
                  "strategy": {"name": "fedavg"}, "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["FedAvg"], f"seed{seed}")})
        cfgs["FedAvg"] = c

        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": "full", "model_hetero": False,
                  "model": {"name": "mock", "embed_choices": [16], "share_encoders": True},
                  "strategy": {"name": "fedprox", "mu": 0.1}, "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["FedProx"], f"seed{seed}")})
        cfgs["FedProx"] = c

    elif scenario == 2:
        # different model, same modality
        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": "full", "model_hetero": True,
                  "model": {"name": "mock", "embed_choices": [16], "share_encoders": False},
                  "strategy": {"name": "fedmd", "public_size": 300, "distill_epochs": 2,
                               "temperature": 2.0},
                  "local": {"local_epochs": 2, "lr": 0.05, "distill_lr": 0.05},
                  "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["FedMD"], f"seed{seed}")})
        cfgs["FedMD"] = c

        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": "full", "model_hetero": True,
                  "model": {"name": "mock", "embed_choices": [16], "share_encoders": False},
                  "strategy": {"name": "fedproto", "proto_mu": 1.0}, "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["FedProto"], f"seed{seed}")})
        cfgs["FedProto"] = c

    elif scenario == 3:
        # same model, different modality subsets (Saha-faithful):
        #   - "mixed:K" -> first K clients multimodal, rest image-only (M:U ratio)
        #   - every client owns the FULL model (all encoders); share everything.
        #     Unimodal clients can't train the text encoder -> their stale copies
        #     dilute the average = the incongruity effect (Saha Sec. 4).
        mixed = f"mixed:{num_multimodal}"
        s3_model = {"name": "mock", "embed_choices": [16],
                    "share_encoders": True, "all_modalities": True}

        # Saha's reference line: fully unimodal FL (all clients image-only).
        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": "uni", "model_hetero": False,
                  "model": dict(s3_model), "strategy": {"name": "fedavg"},
                  "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["UniFL"], f"seed{seed}")})
        cfgs["UniFL"] = c

        # incongruent MMFL baseline (plain FedAvg)
        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": mixed, "model_hetero": False,
                  "model": dict(s3_model), "strategy": {"name": "fedavg"},
                  "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["FedAvg"], f"seed{seed}")})
        cfgs["FedAvg"] = c

        # LOOT (Saha Sec. 7.2): strongest server-level method in the paper
        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": mixed, "model_hetero": False,
                  "model": dict(s3_model),
                  "strategy": {"name": "loot", "public_size": 300,
                               "align_epochs": 1, "align_lr": 0.01},
                  "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["LOOT"], f"seed{seed}")})
        cfgs["LOOT"] = c

        # MIN-lite (Saha Sec. 6): pre-FL feature-level modality imputation
        c = base_cfg(seed, alpha, rounds)
        c.update({"modality_mode": mixed, "model_hetero": False,
                  "model": dict(s3_model), "strategy": {"name": "fedavg"},
                  "min_imputation": True, "with_baselines": True,
                  "run_dir": os.path.join(run_dirs["MIN+FedAvg"], f"seed{seed}")})
        cfgs["MIN+FedAvg"] = c

    else:
        raise SystemExit(f"scenario {scenario} not wired yet (4 comes after MTG)")

    return cfgs


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def mean_std(values: List[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "  n/a"
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.3f}±{s:.3f}"


def mean_or_never(values: List[Any]) -> str:
    vals = [v for v in values if v is not None]
    return "never" if not vals else f"{statistics.mean(vals):.1f}"


def kb(values: List[Any]) -> str:
    vals = [v for v in values if v is not None]
    return "n/a" if not vals else f"{statistics.mean(vals)/1e3:.0f}KB"


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--num-multimodal", type=int, default=2,
                    help="scenario 3: number of multimodal clients (M in M:U)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="Dirichlet alpha (smaller = more non-IID)")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.8,
                    help="accuracy threshold for rounds/comms-to-target")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip running; just (re)draw plots from existing CSVs")
    args = ap.parse_args()

    if args.plot_only:
        store.plot_scenario_summary(args.scenario)
        store.plot_cross_scenario_tax()
        store.plot_scenario_comparison((1, 2))
        print("plots regenerated.")
        return

    ts = store.timestamp()
    seeds = list(range(args.seeds))

    # one timestamped run-dir per method (seeds go inside)
    method_names = {1: ["FedAvg", "FedProx"], 2: ["FedMD", "FedProto"],
                3: ["UniFL", "FedAvg", "LOOT", "MIN+FedAvg"]}[args.scenario]
    run_dirs = {m: store.make_run_dir(args.scenario, m, args.alpha, ts)
                for m in method_names}

    rows: Dict[str, List[Dict[str, Any]]] = {}
    mod_modes: Dict[str, str] = {}
    for seed in seeds:
        cfgs = make_configs(args.scenario, seed, args.alpha, args.rounds,
                            run_dirs, num_multimodal=args.num_multimodal)
        for name, cfg in cfgs.items():
            mod_modes[name] = cfg.get("modality_mode", "full")
            summary = run_experiment(cfg)
            cmods = {int(k): v for k, v in summary.get("client_modalities", {}).items()}
            row = summarize_run(summary, metric="accuracy", target=args.target,
                                client_modalities=cmods)
            rows.setdefault(name, []).append(row)
        print(f"  [seed {seed} done]")

    # ---- screen table + persist aggregated rows ----
    print("\n" + "=" * 104)
    print(f"SCENARIO {args.scenario}  |  alpha={args.alpha}  rounds={args.rounds}  "
          f"seeds={len(seeds)}  target_acc={args.target}")
    print("=" * 104)
    header = (f"{'method':<12}{'final_acc':<14}{'macro_f1':<14}{'worst':<13}"
              f"{'std':<13}{'rnds→tgt':<10}{'comms→tgt':<11}{'AUC':<13}"
              f"{'gap_filled':<12}{'tot_comms':<10}")
    print(header)
    print("-" * 104)
    for name, rs in rows.items():
        print(
            f"{name:<12}"
            f"{mean_std([r['final_accuracy'] for r in rs]):<14}"
            f"{mean_std([r['final_macro_f1'] for r in rs]):<14}"
            f"{mean_std([r['per_client_worst'] for r in rs]):<13}"
            f"{mean_std([r['per_client_std'] for r in rs]):<13}"
            f"{mean_or_never([r['rounds_to_target'] for r in rs]):<10}"
            f"{kb([r['comms_to_target_bytes'] for r in rs]):<11}"
            f"{mean_std([r['auc_learning'] for r in rs]):<13}"
            f"{mean_std([r['gap_filled'] for r in rs]):<12}"
            f"{kb([r['total_bytes'] for r in rs]):<10}"
        )
        # aggregated row -> CSV (mean over seeds)
        agg = {
            "timestamp": ts, "alpha": args.alpha, "rounds": args.rounds,
            "num_clients": 6, "modality_mode": mod_modes.get(name, "full"),
            "seeds": len(seeds),
            "final_accuracy": _mean([r["final_accuracy"] for r in rs]),
            "final_macro_f1": _mean([r["final_macro_f1"] for r in rs]),
            "per_client_worst": _mean([r["per_client_worst"] for r in rs]),
            "per_client_std": _mean([r["per_client_std"] for r in rs]),
            "rounds_to_target": _mean([r["rounds_to_target"] for r in rs]),
            "comms_to_target_bytes": _mean([r["comms_to_target_bytes"] for r in rs]),
            "auc_learning": _mean([r["auc_learning"] for r in rs]),
            "gap_filled": _mean([r["gap_filled"] for r in rs]),
            "total_bytes": _mean([r["total_bytes"] for r in rs]),
            "target": args.target, "run_dir": run_dirs[name],
        }
        store.record_aggregated_run(args.scenario, name, agg)
    print("-" * 104)

    # ---- visualization ----
    store.plot_scenario_summary(args.scenario)
    store.plot_cross_scenario_tax()
    store.plot_scenario_comparison((1, 2))

    print(f"\nSaved: results/scenario{args.scenario}/  "
          f"(runs/, scenario{args.scenario}_summary.csv, plots/)")
    print("Cross-scenario: results/all_runs_summary.csv, results/heterogeneity_tax.png")


if __name__ == "__main__":
    main()
