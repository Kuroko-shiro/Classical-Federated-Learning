"""Result bookkeeping: persist runs, append to per-scenario CSV, and visualize.

Design goals (from the user's requirements):
  - Never overwrite. Every run is saved under a timestamped folder.
  - Group BY SCENARIO so within-scenario comparison is easy
    (results/scenarioN/runs/... + results/scenarioN/scenarioN_summary.csv).
  - Also keep a cross-scenario CSV (results/all_runs_summary.csv) so the
    heterogeneity-tax comparison (S1 -> S2 -> ...) is still one table.
  - Visualize: auto-generate PNGs (e.g. gap_filled vs alpha) per scenario.

All functions are pure-ish helpers operating on the summary rows produced by
metrics.analysis.summarize_run, so scenarios 3/4 reuse them unchanged.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# project root = .../qfl-benchmark ; this file is at src/qflbench/metrics/results_store.py
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
RESULTS_ROOT = os.path.join(_PROJECT_ROOT, "results")

# Columns saved to CSV (stable schema across scenarios 1-4 and later QFL runs).
CSV_FIELDS = [
    "timestamp", "scenario", "method", "alpha", "rounds", "num_clients",
    "modality_mode", "seeds",
    "final_accuracy", "final_macro_f1", "per_client_worst", "per_client_std",
    "rounds_to_target", "comms_to_target_bytes", "auc_learning",
    "gap_filled", "total_bytes", "target", "run_dir",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def scenario_dir(scenario: int) -> str:
    d = os.path.join(RESULTS_ROOT, f"scenario{scenario}")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    return d


def make_run_dir(scenario: int, method: str, alpha: float, ts: str) -> str:
    """Unique, timestamped per-run directory -> never overwritten."""
    safe_method = method.replace(" ", "").replace(":", "")
    d = os.path.join(scenario_dir(scenario), "runs",
                     f"{safe_method}_alpha{alpha}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _append_csv(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    # only keep known fields, in order
    clean = {k: row.get(k, "") for k in CSV_FIELDS}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(clean)


def record_aggregated_run(
    scenario: int,
    method: str,
    agg_row: Dict[str, Any],
) -> None:
    """Append one aggregated (mean-over-seeds) row to BOTH the per-scenario CSV
    and the cross-scenario CSV. Called once per (scenario, method, condition)."""
    agg_row = dict(agg_row)
    agg_row["scenario"] = scenario
    agg_row["method"] = method

    per_scenario_csv = os.path.join(scenario_dir(scenario),
                                    f"scenario{scenario}_summary.csv")
    _append_csv(per_scenario_csv, agg_row)

    cross_csv = os.path.join(RESULTS_ROOT, "all_runs_summary.csv")
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    _append_csv(cross_csv, agg_row)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def plot_scenario_summary(scenario: int) -> Optional[str]:
    """Generate PNGs for a scenario from its accumulated summary CSV.

    Produces, where data allows:
      - gap_filled vs alpha (one line per method)
      - final_accuracy vs alpha (one line per method)
    Saves under results/scenarioN/plots/. Returns the plots dir or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception:
        print("  [viz] matplotlib unavailable; skipping plots")
        return None

    csv_path = os.path.join(scenario_dir(scenario),
                            f"scenario{scenario}_summary.csv")
    rows = _read_csv(csv_path)
    if not rows:
        return None

    plots_dir = os.path.join(scenario_dir(scenario), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # group rows by method
    methods: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        methods.setdefault(r["method"], []).append(r)

    # Primary plots (robust, already correct):
    #   - rounds_to_target vs alpha : the convergence-cost story (MAIN result)
    #   - final_accuracy vs alpha   : absolute performance
    # gap_filled is intentionally NOT plotted on the synthetic task: its
    # denominator (ceiling - floor) collapses here, so values are either guarded
    # out or trivially zero (FedProto), which would mislead. The metric/column is
    # KEPT in analysis.py and the CSV; re-enable plotting on a harder dataset
    # (IU X-ray) where the ceiling-floor gap is real by adding it back here.
    plot_specs = [
        ("rounds_to_target", "rounds to reach target acc"),
        ("final_accuracy", "final accuracy"),
    ]
    for metric, ylabel in plot_specs:
        plt.figure(figsize=(7, 5))
        plotted = False
        for method, rs in methods.items():
            pts = []
            for r in rs:
                a = _to_float(r.get("alpha"))
                v = _to_float(r.get(metric))
                # gap_filled sanity guard: a fraction must lie in [0, 1.05].
                # Older CSV rows may contain pre-guard garbage (e.g. -14, +3);
                # drop them so stale values never reach the plot.
                if metric == "gap_filled" and v is not None and not (-0.01 <= v <= 1.05):
                    v = None
                if a is not None and v is not None:
                    pts.append((a, v))
            if not pts:
                continue
            # average duplicates at the same alpha, then sort
            byx: Dict[float, List[float]] = {}
            for a, v in pts:
                byx.setdefault(a, []).append(v)
            xs = sorted(byx)
            ys = [sum(byx[x]) / len(byx[x]) for x in xs]
            # need >=2 points to draw a meaningful line; a lone surviving point
            # (e.g. gap_filled valid at only one alpha after guarding) is dropped.
            if len(xs) < 2:
                continue
            plt.plot(xs, ys, marker="o", label=method)
            plotted = True
        if not plotted:
            plt.close()
            print(f"  [viz] skipped {metric} (insufficient valid points)")
            continue
        plt.xlabel("Dirichlet α  (smaller = more non-IID)")
        plt.ylabel(ylabel)
        plt.title(f"Scenario {scenario}: {ylabel} vs heterogeneity")
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(plots_dir, f"scenario{scenario}_{metric}_vs_alpha.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [viz] saved {out}")

    return plots_dir


def plot_cross_scenario_tax() -> Optional[str]:
    """Plot the heterogeneity tax across scenarios from the cross-scenario CSV.

    Uses final_accuracy (robust) rather than gap_filled (which self-nulls on easy
    tasks). Draws final_accuracy vs alpha with one line per (scenario, method), so
    the VERTICAL gap between scenario-1 lines and scenario-2 lines at a given alpha
    is the model-heterogeneity tax, and each line's slope shows the non-IID tax."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = _read_csv(os.path.join(RESULTS_ROOT, "all_runs_summary.csv"))
    if not rows:
        return None

    # group by (scenario, method)
    series: Dict[str, List] = {}
    for r in rows:
        a = _to_float(r.get("alpha"))
        v = _to_float(r.get("final_accuracy"))
        if a is None or v is None:
            continue
        key = f"S{r['scenario']}: {r['method']}"
        series.setdefault(key, []).append((a, v))
    if not series:
        return None

    plt.figure(figsize=(8, 5.5))
    for key in sorted(series):
        byx: Dict[float, List[float]] = {}
        for a, v in series[key]:
            byx.setdefault(a, []).append(v)
        xs = sorted(byx)
        ys = [sum(byx[x]) / len(byx[x]) for x in xs]
        plt.plot(xs, ys, marker="o", label=key)
    plt.xlabel("Dirichlet α  (smaller = more non-IID)")
    plt.ylabel("final accuracy")
    plt.title("Heterogeneity tax: accuracy vs non-IID, by scenario/method\n"
              "(vertical gap S1↔S2 = model-heterogeneity tax; slope = non-IID tax)")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    out = os.path.join(RESULTS_ROOT, "heterogeneity_tax.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [viz] saved {out}")
    return out


# --------------------------------------------------------------------------- #
# Cross-scenario comparison plots (case A: one figure per metric, S1 & S2 overlaid)
# --------------------------------------------------------------------------- #
def _collect_series(scenarios):
    """Read each scenario's summary CSV and return rows tagged with scenario.
    Returns {scenario: [rows...]} for scenarios that have a CSV."""
    out = {}
    for sc in scenarios:
        path = os.path.join(scenario_dir(sc), f"scenario{sc}_summary.csv")
        rows = _read_csv(path)
        if rows:
            out[sc] = rows
    return out


def plot_scenario_comparison(scenarios=(1, 2),
                             exclude_methods=("FedProto",)) -> Optional[str]:
    """CASE A: for each key metric, draw ONE figure overlaying all scenarios &
    methods, so S1 vs S2 is directly visible on the same axes.

    Research-optimized choices:
      - `exclude_methods`: drop non-functional CONTRAST methods (FedProto) from the
        comparison. The comparison figure should show the tax that REMAINS when each
        scenario uses its BEST-WORKING method (S1: FedAvg/FedProx, S2: FedMD).
        FedProto's failure is documented separately (per-scenario plots + table), so
        it stays out of the headline comparison where it would only add a confusing
        second green line and suggest "all of S2 fails".
      - metrics: convergence cost (rounds_to_target), accuracy (final_accuracy), and
        fairness (per_client_worst) — the three axes that matter for the medical-FL
        story and that QFL will later be measured against.
      - color = scenario (the model-heterogeneity contrast); line style = method.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [viz] matplotlib unavailable; skipping comparison plots")
        return None

    data = _collect_series(scenarios)
    if len(data) < 1:
        return None

    comp_dir = os.path.join(RESULTS_ROOT, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    scen_colors = {1: "tab:blue", 2: "tab:green", 3: "tab:red", 4: "tab:purple"}
    method_styles = ["-", "--", ":", "-."]
    exclude = set(exclude_methods or ())

    specs = [
        ("rounds_to_target", "rounds to reach target acc",
         "Convergence cost: rounds to target (lower = better)", False),
        ("final_accuracy", "final accuracy",
         "Final accuracy (higher = better)", True),
        ("per_client_worst", "worst-client accuracy",
         "Fairness: worst-client accuracy (higher = better)", True),
    ]

    saved = []
    for metric, ylabel, title, acc_like in specs:
        plt.figure(figsize=(8, 5.5))
        any_line = False
        for sc in sorted(data):
            methods = []
            for r in data[sc]:
                if r["method"] in exclude:
                    continue
                if r["method"] not in methods:
                    methods.append(r["method"])
            for mi, method in enumerate(methods):
                pts = []
                for r in data[sc]:
                    if r["method"] != method:
                        continue
                    a = _to_float(r.get("alpha"))
                    v = _to_float(r.get(metric))
                    if a is not None and v is not None:
                        pts.append((a, v))
                if len(pts) < 1:
                    continue
                byx = {}
                for a, v in pts:
                    byx.setdefault(a, []).append(v)
                xs = sorted(byx)
                ys = [sum(byx[x]) / len(byx[x]) for x in xs]
                plt.plot(xs, ys,
                         color=scen_colors.get(sc, None),
                         linestyle=method_styles[mi % len(method_styles)],
                         marker="o", linewidth=2,
                         label=f"S{sc}: {method}")
                any_line = True
        if not any_line:
            plt.close()
            continue
        plt.xlabel("Dirichlet α  (smaller = more non-IID)")
        plt.ylabel(ylabel)
        plt.title(f"S1 vs S2 — {title}\n(color = scenario, line style = method)")
        plt.xscale("log")
        if acc_like:
            plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        out = os.path.join(comp_dir, f"comparison_{metric}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [viz] saved {out}")
        saved.append(out)

    return comp_dir if saved else None
