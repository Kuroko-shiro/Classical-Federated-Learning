#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = [101, 202, 303]
ORIGINAL = ["ecg_only", "pleth_only", "ecg_pleth"]
CONTROLS = ["ecg_dual", "pleth_dual"]


def load_metrics(root: Path, condition: str, seed: int):
    p = root / f"{condition}_seed{seed}" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_pred(root: Path, condition: str, seed: int):
    p = root / f"{condition}_seed{seed}" / "test_predictions.csv"
    return pd.read_csv(p)


def cluster_bootstrap_delta(subjects, y, p_a, p_b, n_boot=1000, seed=20260905):
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    unique = np.unique(subjects)
    idx_by_subject = {s: np.flatnonzero(subjects == s) for s in unique}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([idx_by_subject[s] for s in sampled])
        yy = y[ids]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(float(average_precision_score(yy, p_a[ids]) - average_precision_score(yy, p_b[ids])))
    a = np.asarray(vals, dtype=float)
    return {
        "n_boot_valid": int(len(a)),
        "mean": float(a.mean()),
        "ci95_low": float(np.quantile(a, 0.025)),
        "ci95_high": float(np.quantile(a, 0.975)),
    }


def mean_seed_cluster_bootstrap(subjects, y, pred_pairs, n_boot=1000, seed=20260905):
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    unique = np.unique(subjects)
    idx_by_subject = {s: np.flatnonzero(subjects == s) for s in unique}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([idx_by_subject[s] for s in sampled])
        yy = y[ids]
        if len(np.unique(yy)) < 2:
            continue
        ds = []
        for pa, pb in pred_pairs:
            ds.append(average_precision_score(yy, pa[ids]) - average_precision_score(yy, pb[ids]))
        vals.append(float(np.mean(ds)))
    a = np.asarray(vals, dtype=float)
    return {
        "n_boot_valid": int(len(a)),
        "mean": float(a.mean()),
        "ci95_low": float(np.quantile(a, 0.025)),
        "ci95_high": float(np.quantile(a, 0.975)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-root", required=True)
    ap.add_argument("--control-root", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    original_root = Path(args.original_root)
    control_root = Path(args.control_root)
    data = Path(args.dataset_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(data / "vitaldb_window_manifest_qc.parquet").reset_index(drop=True)
    test = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    tensor = np.load(data / "windows_qc_f16.npy", mmap_mode="r")

    integrity = {
        "subject_split_disjoint": bool((manifest.groupby("subjectid")["split"].nunique() <= 1).all()),
        "input_not_after_prediction": bool((manifest["input_end"] <= manifest["prediction_start"] + 1e-9).all()),
        "input_prediction_gap_sec_min": float((manifest["prediction_start"] - manifest["input_end"]).min()),
        "input_prediction_gap_sec_max": float((manifest["prediction_start"] - manifest["input_end"]).max()),
        "prediction_horizon_sec_min": float((manifest["prediction_end"] - manifest["prediction_start"]).min()),
        "prediction_horizon_sec_max": float((manifest["prediction_end"] - manifest["prediction_start"]).max()),
        "tensor_n_windows_matches_manifest": bool(tensor.shape[0] == len(manifest)),
        "tensor_two_channels": bool(tensor.ndim == 3 and tensor.shape[1] == 2),
        "tensor_2500_samples": bool(tensor.ndim == 3 and tensor.shape[2] == 2500),
        "n_test_windows": int(len(test)),
        "n_test_subjects": int(test.subjectid.nunique()),
    }

    rows = []
    preds = {}
    alignment_ok = True
    for seed in SEEDS:
        for cond in ORIGINAL:
            m = load_metrics(original_root, cond, seed)
            pr = load_pred(original_root, cond, seed)
            if len(pr) != len(test) or not np.array_equal(pr["y"].to_numpy().astype(int), test["label"].to_numpy().astype(int)):
                alignment_ok = False
            preds[(cond, seed)] = pr["p"].to_numpy(dtype=float)
            rows.append({"family": "original", "condition": cond, "seed": seed, "n_params": int(m["n_params"]),
                         "auprc": float(m["test"]["auprc"]), "auroc": float(m["test"]["auroc"]), "f1": float(m["test"]["f1"])})
        for cond in CONTROLS:
            m = load_metrics(control_root, cond, seed)
            pr = load_pred(control_root, cond, seed)
            if len(pr) != len(test) or not np.array_equal(pr["y"].to_numpy().astype(int), test["label"].to_numpy().astype(int)):
                alignment_ok = False
            preds[(cond, seed)] = pr["p"].to_numpy(dtype=float)
            rows.append({"family": "capacity_control", "condition": cond, "seed": seed, "n_params": int(m["n_params"]),
                         "auprc": float(m["test"]["auprc"]), "auroc": float(m["test"]["auroc"]), "f1": float(m["test"]["f1"])})
    integrity["prediction_label_alignment"] = bool(alignment_ok)
    integrity_pass = all([
        integrity["subject_split_disjoint"], integrity["input_not_after_prediction"],
        integrity["tensor_n_windows_matches_manifest"], integrity["tensor_two_channels"],
        integrity["tensor_2500_samples"], integrity["prediction_label_alignment"]
    ])
    integrity["overall_pass"] = bool(integrity_pass)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "vitaldb_gate0_capacity_metrics.csv", index=False)

    comp_rows = []
    y = test["label"].to_numpy(dtype=int)
    subj = test["subjectid"].to_numpy()
    for seed in SEEDS:
        fusion = preds[("ecg_pleth", seed)]
        ecg_dual = preds[("ecg_dual", seed)]
        pleth_dual = preds[("pleth_dual", seed)]
        ap_f = average_precision_score(y, fusion)
        ap_e = average_precision_score(y, ecg_dual)
        ap_p = average_precision_score(y, pleth_dual)
        b_e = cluster_bootstrap_delta(subj, y, fusion, ecg_dual, args.n_boot, 9000 + seed)
        b_p = cluster_bootstrap_delta(subj, y, fusion, pleth_dual, args.n_boot, 12000 + seed)
        comp_rows.append({
            "seed": seed,
            "fusion_auprc": float(ap_f),
            "ecg_dual_auprc": float(ap_e),
            "pleth_dual_auprc": float(ap_p),
            "best_dual_auprc": float(max(ap_e, ap_p)),
            "fusion_minus_ecg_dual": float(ap_f - ap_e),
            "fusion_minus_pleth_dual": float(ap_f - ap_p),
            "fusion_minus_best_dual": float(ap_f - max(ap_e, ap_p)),
            "fusion_minus_ecg_dual_ci95_low": b_e["ci95_low"],
            "fusion_minus_ecg_dual_ci95_high": b_e["ci95_high"],
            "fusion_minus_pleth_dual_ci95_low": b_p["ci95_low"],
            "fusion_minus_pleth_dual_ci95_high": b_p["ci95_high"],
        })
    comps = pd.DataFrame(comp_rows)
    comps.to_csv(out / "vitaldb_gate0_capacity_comparison_by_seed.csv", index=False)

    pooled_ecg = mean_seed_cluster_bootstrap(
        subj, y, [(preds[("ecg_pleth", s)], preds[("ecg_dual", s)]) for s in SEEDS],
        args.n_boot, 2026090501)
    pooled_pleth = mean_seed_cluster_bootstrap(
        subj, y, [(preds[("ecg_pleth", s)], preds[("pleth_dual", s)]) for s in SEEDS],
        args.n_boot, 2026090502)

    by_cond = metrics.groupby("condition").agg(
        mean_auprc=("auprc", "mean"), sd_auprc=("auprc", "std"),
        mean_auroc=("auroc", "mean"), sd_auroc=("auroc", "std"),
        mean_f1=("f1", "mean"), n_params=("n_params", "first")
    ).reset_index()
    by_cond.to_csv(out / "vitaldb_gate0_capacity_summary_by_condition.csv", index=False)

    mean_best_dual = float(comps["best_dual_auprc"].mean())
    mean_fusion = float(comps["fusion_auprc"].mean())
    gain_vs_best_dual = float((comps["fusion_auprc"] - comps["best_dual_auprc"]).mean())
    positive_vs_best_dual = int((comps["fusion_minus_best_dual"] > 0).sum())

    if not integrity_pass:
        interpretation = "STOP_INTEGRITY_FAILURE"
    elif gain_vs_best_dual <= 0 or positive_vs_best_dual < 2:
        interpretation = "YELLOW_CAPACITY_CAN_EXPLAIN_FUSION_GAIN"
    else:
        interpretation = "YELLOW_COMPLEMENTARITY_SUPPORTED_BUT_BELOW_PREREGISTERED_0.02_TARGET"

    summary = {
        "canonical_gate0_decision_remains": "YELLOW",
        "diagnostic_interpretation": interpretation,
        "reason_gate_cannot_be_upgraded": "The preregistered practical Strong-GO target was mean AUPRC fusion gain >=0.02; observed original gain was +0.0133.",
        "integrity": integrity,
        "fusion_mean_auprc": mean_fusion,
        "best_capacity_matched_single_sensor_mean_auprc": mean_best_dual,
        "fusion_gain_vs_best_capacity_matched_single_sensor_mean": gain_vs_best_dual,
        "positive_vs_best_capacity_matched_single_sensor_seeds": positive_vs_best_dual,
        "cluster_bootstrap_mean_seed_delta_vs_ecg_dual": pooled_ecg,
        "cluster_bootstrap_mean_seed_delta_vs_pleth_dual": pooled_pleth,
        "notes": [
            "ecg_dual and pleth_dual use exactly 55,681 parameters, the same two-encoder architecture and fusion head as ECG+PLETH.",
            "Each capacity control duplicates one selected sensor into both branches, isolating extra model capacity from cross-sensor information.",
            "Bootstrap resamples subjects, not windows, to respect within-subject clustering.",
            "The capacity diagnostic does not retroactively change the preregistered Strong-GO threshold."
        ]
    }
    (out / "vitaldb_gate0_capacity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# VitalDB Gate 0 — Yellow capacity diagnostic",
        "",
        f"Canonical decision remains: **YELLOW**",
        f"Diagnostic interpretation: **{interpretation}**",
        "",
        "## Integrity audit",
        "",
    ]
    for k, v in integrity.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Capacity-matched results", "",
              "| condition | AUPRC mean±sd | AUROC mean±sd | params |",
              "|---|---:|---:|---:|"]
    for r in by_cond.itertuples(index=False):
        lines.append(f"| {r.condition} | {r.mean_auprc:.4f} ± {r.sd_auprc:.4f} | {r.mean_auroc:.4f} ± {r.sd_auroc:.4f} | {int(r.n_params):,} |")
    lines += [
        "",
        f"Fusion AUPRC mean: **{mean_fusion:.4f}**",
        f"Best capacity-matched single-sensor AUPRC mean: **{mean_best_dual:.4f}**",
        f"Mean fusion gain vs best capacity-matched control: **{gain_vs_best_dual:+.4f}**",
        f"Positive seeds vs best capacity-matched control: **{positive_vs_best_dual}/3**",
        "",
        "Subject-clustered bootstrap (mean across the three fixed seeds):",
        f"- fusion − ECG-dual: {pooled_ecg['mean']:+.4f} (95% CI {pooled_ecg['ci95_low']:+.4f}, {pooled_ecg['ci95_high']:+.4f})",
        f"- fusion − PLETH-dual: {pooled_pleth['mean']:+.4f} (95% CI {pooled_pleth['ci95_low']:+.4f}, {pooled_pleth['ci95_high']:+.4f})",
        "",
        "The preregistered Strong-GO target (+0.02 AUPRC over the best ordinary unimodal model) is not changed after observing the result. Therefore this diagnostic can explain the Yellow result, but cannot upgrade it to Strong GO by itself.",
    ]
    (out / "vitaldb_gate0_capacity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
