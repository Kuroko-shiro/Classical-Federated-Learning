#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

ART = "SNUADC/ART"
FS = 125.0
N_SAMPLES = 2500
MIN_FINITE = 0.95
MAX_NAN_RUN_SAMPLES = int(0.5 * FS)
MAX_FLAT_DIFF = 0.98
MAX_ROBUST_OUTLIER = 0.20
ART_MIN_P01_P99_RANGE = 5.0
CLIP_SIGMA = 20.0


def max_true_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    x = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(x)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max()) if len(starts) else 0


def art_qc_and_normalize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    finite_fraction = float(finite.mean())
    nan_run = max_true_run(~finite)
    rec = {"finite_fraction": finite_fraction, "max_nan_run_samples": nan_run}
    if finite_fraction < MIN_FINITE:
        return None, rec, "finite_fraction"
    if nan_run > MAX_NAN_RUN_SAMPLES:
        return None, rec, "long_nan_run"

    idx = np.arange(len(x))
    y = x.copy()
    if not finite.all():
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    q01, q25, q50, q75, q99 = np.quantile(y, [0.01, 0.25, 0.50, 0.75, 0.99])
    mad = float(np.median(np.abs(y - q50)))
    iqr = float(q75 - q25)
    robust_scale = max(mad * 1.4826, iqr / 1.349, 1e-8)
    robust_range = float(q99 - q01)
    flat_fraction = float(np.mean(np.abs(np.diff(y)) < 1e-8)) if len(y) > 1 else 1.0
    outlier_fraction = float(np.mean(np.abs(y - q50) > CLIP_SIGMA * robust_scale))
    rec.update({
        "p01": float(q01), "p50": float(q50), "p99": float(q99),
        "robust_range": robust_range, "robust_scale": float(robust_scale),
        "flat_diff_fraction": flat_fraction, "robust_outlier_fraction": outlier_fraction,
    })
    if robust_range < ART_MIN_P01_P99_RANGE or robust_scale <= 1e-7:
        return None, rec, "low_dynamic_range"
    if flat_fraction > MAX_FLAT_DIFF:
        return None, rec, "flatline"
    if outlier_fraction > MAX_ROBUST_OUTLIER:
        return None, rec, "artifact_fraction"

    lo = q50 - CLIP_SIGMA * robust_scale
    hi = q50 + CLIP_SIGMA * robust_scale
    y = np.clip(y, lo, hi)
    y = (y - q50) / robust_scale
    y = np.clip(y, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float16)
    return y, rec, "ok"


def extract_case(caseid: int, g: pd.DataFrame):
    st = time.time()
    rows = []
    try:
        arr = np.asarray(vitaldb.load_case(caseid, [ART], interval=1.0 / FS)).reshape(-1)
        n = len(arr)
        for r in g.itertuples(index=False):
            i0 = int(round(float(r.input_start) * FS))
            i1 = i0 + N_SAMPLES
            base = {"source_index": int(r.source_index), "caseid": caseid, "window_id": str(r.window_id)}
            if i0 < 0 or i1 > n:
                rows.append((int(r.source_index), None, {**base, "art_qc_pass": False, "reject_reason": "bounds"}))
                continue
            art, rec, reason = art_qc_and_normalize(arr[i0:i1])
            q = {**base, **{f"art_{k}": v for k, v in rec.items()}}
            if art is None:
                q.update(art_qc_pass=False, reject_reason=reason)
                rows.append((int(r.source_index), None, q))
            else:
                q.update(art_qc_pass=True, reject_reason="ok")
                rows.append((int(r.source_index), art, q))
        return rows, {"caseid": caseid, "ok": True, "load_seconds": float(time.time()-st), "n_case_samples": int(n)}
    except Exception as e:
        for r in g.itertuples(index=False):
            rows.append((int(r.source_index), None, {"source_index": int(r.source_index), "caseid": caseid,
                         "window_id": str(r.window_id), "art_qc_pass": False,
                         "reject_reason": f"case_load:{e!r}"}))
        return rows, {"caseid": caseid, "ok": False, "load_seconds": float(time.time()-st), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.p1_data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(src / "vitaldb_window_manifest_qc.parquet").reset_index(drop=True)
    x_p1 = np.load(src / "windows_qc_f16.npy", mmap_mode="r")
    if len(m) != x_p1.shape[0] or x_p1.shape[1:] != (2, N_SAMPLES):
        raise RuntimeError(f"P1 tensor/manifest mismatch: manifest={len(m)}, tensor={x_p1.shape}")
    m["source_index"] = np.arange(len(m), dtype=np.int64)

    art_temp_path = out / "art_all_temp_f16.npy"
    art_mm = np.lib.format.open_memmap(art_temp_path, mode="w+", dtype=np.float16,
                                       shape=(len(m), N_SAMPLES))
    art_mm[:] = 0
    qc_rows, case_rows = [], []
    groups = [(int(cid), g.copy()) for cid, g in m.groupby("caseid", sort=False)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_case, cid, g): cid for cid, g in groups}
        for i, fut in enumerate(as_completed(futs), 1):
            items, cstat = fut.result()
            case_rows.append(cstat)
            for idx, art, q in items:
                qc_rows.append(q)
                if art is not None:
                    art_mm[idx] = art
            if i % 200 == 0 or i == len(futs):
                art_mm.flush()
                print(f"ART extracted {i}/{len(futs)} P1-QC cases", flush=True)
    art_mm.flush()

    qc = pd.DataFrame(qc_rows).sort_values("source_index").reset_index(drop=True)
    cases = pd.DataFrame(case_rows).sort_values("caseid")
    if len(qc) != len(m):
        raise RuntimeError(f"ART QC rows {len(qc)} != P1 manifest {len(m)}")
    keep = qc["art_qc_pass"].fillna(False).to_numpy(bool)
    ids = qc.loc[keep, "source_index"].to_numpy(np.int64)

    final_path = out / "windows_ecg_art_qc_f16.npy"
    final = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float16,
                                      shape=(len(ids), 2, N_SAMPLES))
    block = 2048
    for j in range(0, len(ids), block):
        sel = ids[j:j+block]
        final[j:j+len(sel), 0] = x_p1[sel, 0]
        final[j:j+len(sel), 1] = art_mm[sel]
    final.flush()
    del final
    del art_mm
    os.remove(art_temp_path)

    final_m = m.loc[keep].copy().reset_index(drop=True)
    final_m["p2_tensor_index"] = np.arange(len(final_m), dtype=np.int64)
    final_m.to_parquet(out / "vitaldb_p2_window_manifest_qc.parquet", index=False)
    final_m.to_csv(out / "vitaldb_p2_window_manifest_qc.csv", index=False)
    qc.to_parquet(out / "vitaldb_art_qc_all.parquet", index=False)
    cases.to_csv(out / "vitaldb_art_case_load_audit.csv", index=False)

    splits=[]
    for sp in ["train","val","test"]:
        s=final_m[final_m.split.eq(sp)]
        splits.append({"split":sp,"n_subjects":int(s.subjectid.nunique()),"n_cases":int(s.caseid.nunique()),
                       "n_windows":int(len(s)),"n_positive":int(s.label.sum()),
                       "prevalence":float(s.label.mean()) if len(s) else None})
    kept_cases=int(final_m.caseid.nunique())
    summary={
        "design":"P2 diagnostic on the canonical P1 ECG+PLETH-QC cohort; ART is added and only ART-QC failures are removed",
        "not_primary_benchmark": True,
        "label_remains":"future Solar8000/ART_MBP hypotension event; ART waveform is diagnostic input only",
        "p1_qc_windows":int(len(m)),
        "p2_qc_windows":int(len(final_m)),
        "p2_qc_fraction_of_p1":float(len(final_m)/len(m)),
        "p2_qc_cases":kept_cases,
        "p2_qc_subjects":int(final_m.subjectid.nunique()),
        "p2_positive_windows":int(final_m.label.sum()),
        "p2_prevalence":float(final_m.label.mean()),
        "scale_gate":"P2_SCALE_GO" if kept_cases>=2500 else ("P2_SCALE_YELLOW" if kept_cases>=1500 else "P2_SCALE_NO_GO"),
        "sampling_rate_hz":FS,
        "window_seconds":20,
        "art_qc_policy":{
            "min_finite_fraction":MIN_FINITE,
            "max_nan_run_sec":MAX_NAN_RUN_SAMPLES/FS,
            "max_flat_diff_fraction":MAX_FLAT_DIFF,
            "max_robust_outlier_fraction_20sigma":MAX_ROBUST_OUTLIER,
            "art_min_p01_p99_range_mmhg":ART_MIN_P01_P99_RANGE,
        },
        "splits":splits,
        "tensor_file":"windows_ecg_art_qc_f16.npy",
        "tensor_shape":[int(len(final_m)),2,N_SAMPLES],
        "tensor_dtype":"float16",
        "top_reject_reasons":qc.loc[~keep,"reject_reason"].value_counts().head(20).to_dict(),
    }
    (out/"vitaldb_p2_qc_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__":
    main()
