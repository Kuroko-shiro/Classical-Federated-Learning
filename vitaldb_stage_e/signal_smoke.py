#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import vitaldb

API = "https://api.vitaldb.net"
ECG = "SNUADC/ECG_II"
PLETH = "SNUADC/PLETH"
MAP = "Solar8000/ART_MBP"
FS = 125.0


def robust_stats(x: np.ndarray, prefix: str):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = {f"{prefix}_finite_fraction": float(finite.mean()) if len(x) else 0.0}
    y = x[finite]
    if len(y) < 2:
        return out
    q = np.quantile(y, [0, .001, .01, .25, .5, .75, .99, .999, 1])
    keys = ["min", "p001", "p01", "p25", "p50", "p75", "p99", "p999", "max"]
    for k, v in zip(keys, q):
        out[f"{prefix}_{k}"] = float(v)
    mad = float(np.median(np.abs(y - q[4])))
    iqr = float(q[5] - q[3])
    out[f"{prefix}_mad"] = mad
    out[f"{prefix}_iqr"] = iqr
    scale = max(mad * 1.4826, iqr / 1.349, 1e-8)
    out[f"{prefix}_robust_outlier_fraction_20sigma"] = float(np.mean(np.abs(y - q[4]) > 20.0 * scale))
    d = np.diff(y)
    out[f"{prefix}_flat_diff_fraction"] = float(np.mean(np.abs(d) < 1e-8)) if len(d) else 1.0
    out[f"{prefix}_robust_range_p01_p99"] = float(q[6] - q[2])
    return out


def load_one(caseid: int):
    rec = {"caseid": caseid}
    st = time.time()
    try:
        arr = vitaldb.load_case(caseid, [ECG, PLETH], interval=1.0 / FS)
        arr = np.asarray(arr)
        rec["load_seconds"] = float(time.time() - st)
        rec["shape_rows"] = int(arr.shape[0]) if arr.ndim == 2 else 0
        rec["shape_cols"] = int(arr.shape[1]) if arr.ndim == 2 else 0
        rec["duration_sec"] = float(arr.shape[0] / FS) if arr.ndim == 2 else 0.0
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"unexpected shape {arr.shape}")
        rec.update(robust_stats(arr[:, 0], "ecg"))
        rec.update(robust_stats(arr[:, 1], "pleth"))
        rec["ok"] = True
    except Exception as e:
        rec["load_seconds"] = float(time.time() - st)
        rec["ok"] = False
        rec["error"] = repr(e)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vitaldb_stage_e_smoke")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(f"{API}/cases")
    trks = pd.read_csv(f"{API}/trks")
    def ids(name):
        return set(trks.loc[trks["tname"].eq(name), "caseid"].astype(int))
    p1 = np.array(sorted(ids(ECG) & ids(PLETH) & ids(MAP)), dtype=int)
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(p1, size=min(args.n, len(p1)), replace=False))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(load_one, int(cid)): int(cid) for cid in chosen}
        for i, fut in enumerate(as_completed(futs), start=1):
            rows.append(fut.result())
            if i % 10 == 0 or i == len(futs):
                print(f"loaded {i}/{len(futs)}", flush=True)
    df = pd.DataFrame(rows).sort_values("caseid")
    df.to_csv(out / "vitaldb_packed_signal_smoke.csv", index=False)

    ok = df[df["ok"].fillna(False)]
    def med(col):
        return float(ok[col].median()) if col in ok and len(ok) else None
    def q95(col):
        return float(ok[col].quantile(.95)) if col in ok and len(ok) else None
    summary = {
        "n_requested": int(len(chosen)),
        "n_success": int(len(ok)),
        "success_fraction": float(len(ok) / len(chosen)) if len(chosen) else 0.0,
        "sampling_rate_hz": FS,
        "median_load_seconds_per_case": med("load_seconds"),
        "median_duration_sec": med("duration_sec"),
        "ecg_median_finite_fraction": med("ecg_finite_fraction"),
        "pleth_median_finite_fraction": med("pleth_finite_fraction"),
        "ecg_median_p01": med("ecg_p01"),
        "ecg_median_p99": med("ecg_p99"),
        "pleth_median_p01": med("pleth_p01"),
        "pleth_median_p99": med("pleth_p99"),
        "ecg_95pct_robust_outlier_fraction": q95("ecg_robust_outlier_fraction_20sigma"),
        "pleth_95pct_robust_outlier_fraction": q95("pleth_robust_outlier_fraction_20sigma"),
        "ecg_cases_max_gt_20": int((ok.get("ecg_max", pd.Series(dtype=float)) > 20).sum()),
        "pleth_cases_max_gt_500": int((ok.get("pleth_max", pd.Series(dtype=float)) > 500).sum()),
        "interpretation": "Use robust percentiles/outlier fractions rather than raw maxima to decide whether large extrema are sparse artifacts. Final QC is window-level, not whole-case-level.",
    }
    (out / "vitaldb_packed_signal_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# VitalDB packed signal smoke QC",
        "",
        f"- requested: **{summary['n_requested']}**",
        f"- success: **{summary['n_success']}**",
        f"- median load time/case: **{summary['median_load_seconds_per_case']:.2f} s**" if summary['median_load_seconds_per_case'] is not None else "- median load time/case: n/a",
        f"- ECG median finite fraction: **{summary['ecg_median_finite_fraction']:.4f}**" if summary['ecg_median_finite_fraction'] is not None else "- ECG finite: n/a",
        f"- PLETH median finite fraction: **{summary['pleth_median_finite_fraction']:.4f}**" if summary['pleth_median_finite_fraction'] is not None else "- PLETH finite: n/a",
        f"- ECG median p1–p99: **{summary['ecg_median_p01']:.4g} to {summary['ecg_median_p99']:.4g}**" if summary['ecg_median_p01'] is not None else "",
        f"- PLETH median p1–p99: **{summary['pleth_median_p01']:.4g} to {summary['pleth_median_p99']:.4g}**" if summary['pleth_median_p01'] is not None else "",
        f"- Cases with raw ECG max >20: **{summary['ecg_cases_max_gt_20']}**",
        f"- Cases with raw PLETH max >500: **{summary['pleth_cases_max_gt_500']}**",
        "",
        "Final QC will be applied to each 20-second model-input window using finite fraction, robust outlier fraction and flatline checks.",
    ]
    (out / "vitaldb_packed_signal_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
