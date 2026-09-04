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

ECG = "SNUADC/ECG_II"
PLETH = "SNUADC/PLETH"
FS = 125.0
N_SAMPLES = 2500
MIN_FINITE = 0.95
MAX_NAN_RUN_SAMPLES = int(0.5 * FS)
MAX_FLAT_DIFF = 0.98
MAX_ROBUST_OUTLIER = 0.20
ECG_MIN_P01_P99_RANGE = 0.05
PLETH_MIN_P01_P99_RANGE = 1.0
CLIP_SIGMA = 20.0


def max_true_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    x = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(x)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max()) if len(starts) else 0


def channel_qc_and_normalize(x: np.ndarray, modality: str):
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

    min_range = ECG_MIN_P01_P99_RANGE if modality == "ecg" else PLETH_MIN_P01_P99_RANGE
    if robust_range < min_range or robust_scale <= 1e-7:
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


def extract_case(caseid: int, case_manifest: pd.DataFrame):
    st = time.time()
    result = []
    try:
        arr = np.asarray(vitaldb.load_case(caseid, [ECG, PLETH], interval=1.0 / FS))
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"unexpected shape {arr.shape}")
        n = len(arr)
        for r in case_manifest.itertuples(index=False):
            i0 = int(round(float(r.input_start) * FS))
            i1 = i0 + N_SAMPLES
            base = {"source_index": int(r.source_index), "caseid": caseid, "window_id": str(r.window_id)}
            if i0 < 0 or i1 > n:
                result.append((int(r.source_index), None, {**base, "signal_qc_pass": False, "reject_reason": "bounds"}))
                continue
            ecg, ecg_rec, ecg_reason = channel_qc_and_normalize(arr[i0:i1, 0], "ecg")
            pleth, pleth_rec, pleth_reason = channel_qc_and_normalize(arr[i0:i1, 1], "pleth")
            q = {**base,
                 **{f"ecg_{k}": v for k, v in ecg_rec.items()},
                 **{f"pleth_{k}": v for k, v in pleth_rec.items()}}
            if ecg is None or pleth is None:
                q.update(signal_qc_pass=False, reject_reason=f"ecg:{ecg_reason}|pleth:{pleth_reason}")
                result.append((int(r.source_index), None, q))
            else:
                x = np.stack([ecg, pleth], axis=0)
                q.update(signal_qc_pass=True, reject_reason="ok")
                result.append((int(r.source_index), x, q))
        return result, {"caseid": caseid, "ok": True, "load_seconds": float(time.time() - st), "n_case_samples": int(n)}
    except Exception as e:
        for r in case_manifest.itertuples(index=False):
            base = {"source_index": int(r.source_index), "caseid": caseid, "window_id": str(r.window_id),
                    "signal_qc_pass": False, "reject_reason": f"case_load:{e!r}"}
            result.append((int(r.source_index), None, base))
        return result, {"caseid": caseid, "ok": False, "load_seconds": float(time.time() - st), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="artifacts/vitaldb_stage_g/dataset")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest).reset_index(drop=True)
    manifest["source_index"] = np.arange(len(manifest), dtype=np.int64)
    temp_path = out / "windows_all_temp_f16.npy"
    mm = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float16,
                                   shape=(len(manifest), 2, N_SAMPLES))
    mm[:] = 0

    qc_rows, case_rows = [], []
    groups = [(int(cid), g.copy()) for cid, g in manifest.groupby("caseid", sort=False)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_case, cid, g): cid for cid, g in groups}
        for i, fut in enumerate(as_completed(futs), 1):
            items, cstat = fut.result()
            case_rows.append(cstat)
            for idx, x, q in items:
                qc_rows.append(q)
                if x is not None:
                    mm[idx] = x
            if i % 200 == 0 or i == len(futs):
                mm.flush()
                print(f"extracted {i}/{len(futs)} cases", flush=True)
    mm.flush()

    qc = pd.DataFrame(qc_rows).sort_values("source_index").reset_index(drop=True)
    cases = pd.DataFrame(case_rows).sort_values("caseid")
    if len(qc) != len(manifest):
        raise RuntimeError(f"QC rows {len(qc)} != manifest {len(manifest)}")
    pass_mask = qc["signal_qc_pass"].fillna(False).to_numpy(bool)
    accepted_src = qc.loc[pass_mask, "source_index"].to_numpy(np.int64)

    final_path = out / "windows_qc_f16.npy"
    final = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float16,
                                      shape=(len(accepted_src), 2, N_SAMPLES))
    block = 2048
    for j in range(0, len(accepted_src), block):
        ids = accepted_src[j:j+block]
        final[j:j+len(ids)] = mm[ids]
    final.flush()
    del final
    del mm
    os.remove(temp_path)

    qc["tensor_index"] = -1
    qc.loc[pass_mask, "tensor_index"] = np.arange(pass_mask.sum(), dtype=np.int64)
    final_manifest = manifest.merge(qc[["source_index","signal_qc_pass","reject_reason","tensor_index"]],
                                    on="source_index", how="left")
    final_manifest = final_manifest[final_manifest.signal_qc_pass.fillna(False)].copy()
    final_manifest.to_parquet(out / "vitaldb_window_manifest_qc.parquet", index=False)
    final_manifest.to_csv(out / "vitaldb_window_manifest_qc.csv", index=False)
    qc.to_parquet(out / "vitaldb_signal_qc_all.parquet", index=False)
    cases.to_csv(out / "vitaldb_signal_case_load_audit.csv", index=False)

    split_rows=[]
    for sp in ["train","val","test"]:
        m=final_manifest[final_manifest.split==sp]
        split_rows.append({"split":sp,"n_subjects":int(m.subjectid.nunique()),"n_cases":int(m.caseid.nunique()),
                           "n_windows":int(len(m)),"n_positive":int(m.label.sum()),"prevalence":float(m.label.mean()) if len(m) else None})
    kept_cases=int(final_manifest.caseid.nunique())
    gate="SIGNAL_QC_SCALE_GO" if kept_cases>=2500 else ("SIGNAL_QC_SCALE_YELLOW" if kept_cases>=1500 else "SIGNAL_QC_SCALE_NO_GO")
    reasons=qc.loc[~pass_mask,"reject_reason"].value_counts().head(20).to_dict()
    summary={
        "input_manifest_windows":int(len(manifest)),
        "qc_pass_windows":int(pass_mask.sum()),
        "qc_pass_fraction":float(pass_mask.mean()),
        "qc_pass_cases":kept_cases,
        "qc_pass_subjects":int(final_manifest.subjectid.nunique()),
        "qc_positive_windows":int(final_manifest.label.sum()),
        "qc_prevalence":float(final_manifest.label.mean()),
        "scale_gate":gate,
        "sampling_rate_hz":FS,
        "window_seconds":20,
        "normalization":"per-window median / robust scale after interpolation of <=0.5 s gaps and +/-20 robust-scale clipping",
        "qc_policy":{
            "min_finite_fraction":MIN_FINITE,
            "max_nan_run_sec":MAX_NAN_RUN_SAMPLES/FS,
            "max_flat_diff_fraction":MAX_FLAT_DIFF,
            "max_robust_outlier_fraction_20sigma":MAX_ROBUST_OUTLIER,
            "ecg_min_p01_p99_range":ECG_MIN_P01_P99_RANGE,
            "pleth_min_p01_p99_range":PLETH_MIN_P01_P99_RANGE,
        },
        "top_reject_reasons":reasons,
        "splits":split_rows,
        "tensor_file":"windows_qc_f16.npy",
        "tensor_shape":[int(pass_mask.sum()),2,N_SAMPLES],
        "tensor_dtype":"float16",
    }
    (out/"vitaldb_signal_qc_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__":
    main()
