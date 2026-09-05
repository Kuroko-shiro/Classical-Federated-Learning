#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

# When this file is executed as `python vitaldb_stage_k/extract_windows_60s.py`,
# Python puts vitaldb_stage_k/ (not the repository root) on sys.path. Add the
# repository root explicitly so the canonical Stage G implementation can be
# reused without packaging the research-stage directories.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vitaldb_stage_g.extract_windows import (
    ECG, PLETH, FS, channel_qc_and_normalize,
    MIN_FINITE, MAX_NAN_RUN_SAMPLES, MAX_FLAT_DIFF, MAX_ROBUST_OUTLIER,
    ECG_MIN_P01_P99_RANGE, PLETH_MIN_P01_P99_RANGE,
)

WINDOW_SECONDS = 60.0
N_SAMPLES = int(FS * WINDOW_SECONDS)


def extract_case(caseid: int, case_manifest: pd.DataFrame):
    st = time.time()
    result = []
    try:
        arr = np.asarray(vitaldb.load_case(caseid, [ECG, PLETH], interval=1.0 / FS))
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"unexpected shape {arr.shape}")
        n = len(arr)
        for r in case_manifest.itertuples(index=False):
            # End-align 60 s history to exactly the same input_end/prediction_start as canonical P1.
            i1 = int(round(float(r.input_end) * FS))
            i0 = i1 - N_SAMPLES
            base = {"source_index": int(r.source_index), "caseid": caseid, "window_id": str(r.window_id)}
            if abs(float(r.input_end) - float(r.prediction_start)) > 1e-6:
                raise ValueError("canonical manifest input_end != prediction_start")
            if i0 < 0 or i1 > n:
                result.append((int(r.source_index), None, {**base, "signal_qc_pass": False, "reject_reason": "bounds_60s"}))
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
                q.update(signal_qc_pass=True, reject_reason="ok")
                result.append((int(r.source_index), np.stack([ecg, pleth], axis=0), q))
        return result, {"caseid": caseid, "ok": True, "load_seconds": float(time.time() - st), "n_case_samples": int(n)}
    except Exception as e:
        for r in case_manifest.itertuples(index=False):
            result.append((int(r.source_index), None, {
                "source_index": int(r.source_index), "caseid": caseid, "window_id": str(r.window_id),
                "signal_qc_pass": False, "reject_reason": f"case_load:{e!r}"}))
        return result, {"caseid": caseid, "ok": False, "load_seconds": float(time.time() - st), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="work60/dataset")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest).reset_index(drop=True)
    manifest["source_index"] = np.arange(len(manifest), dtype=np.int64)
    if not np.allclose(manifest["input_end"].to_numpy(float), manifest["prediction_start"].to_numpy(float)):
        raise RuntimeError("P1 manifest is not end-aligned to prediction start")

    temp_path = out / "windows_all_temp_f16.npy"
    mm = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float16,
                                   shape=(len(manifest), 2, N_SAMPLES))
    mm[:] = 0
    qc_rows, case_rows = [], []
    groups = [(int(cid), g.copy()) for cid, g in manifest.groupby("caseid", sort=False)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_case, cid, g): cid for cid, g in groups}
        for i, fut in enumerate(as_completed(futs), 1):
            items, cstat = fut.result(); case_rows.append(cstat)
            for idx, x, q in items:
                qc_rows.append(q)
                if x is not None: mm[idx] = x
            if i % 200 == 0 or i == len(futs):
                mm.flush(); print(f"extracted {i}/{len(futs)} cases", flush=True)
    mm.flush()

    qc = pd.DataFrame(qc_rows).sort_values("source_index").reset_index(drop=True)
    cases = pd.DataFrame(case_rows).sort_values("caseid")
    if len(qc) != len(manifest): raise RuntimeError(f"QC rows {len(qc)} != manifest {len(manifest)}")
    pass_mask = qc["signal_qc_pass"].fillna(False).to_numpy(bool)
    accepted_src = qc.loc[pass_mask, "source_index"].to_numpy(np.int64)

    final_path = out / "windows_qc_f16.npy"
    final = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float16,
                                      shape=(len(accepted_src), 2, N_SAMPLES))
    block = 512
    for j in range(0, len(accepted_src), block):
        ids = accepted_src[j:j+block]; final[j:j+len(ids)] = mm[ids]
    final.flush(); del final; del mm; os.remove(temp_path)

    qc["tensor_index"] = -1
    qc.loc[pass_mask, "tensor_index"] = np.arange(pass_mask.sum(), dtype=np.int64)
    final_manifest = manifest.merge(qc[["source_index","signal_qc_pass","reject_reason","tensor_index"]], on="source_index", how="left")
    final_manifest = final_manifest[final_manifest.signal_qc_pass.fillna(False)].copy()
    final_manifest.to_parquet(out / "vitaldb_window_manifest_qc.parquet", index=False)
    qc.to_parquet(out / "vitaldb_signal_qc_all.parquet", index=False)
    cases.to_csv(out / "vitaldb_signal_case_load_audit.csv", index=False)

    split_rows = []
    for sp in ["train","val","test"]:
        m = final_manifest[final_manifest.split == sp]
        split_rows.append({"split":sp,"n_subjects":int(m.subjectid.nunique()),"n_cases":int(m.caseid.nunique()),
                           "n_windows":int(len(m)),"n_positive":int(m.label.sum()),"prevalence":float(m.label.mean()) if len(m) else None})
    kept_cases = int(final_manifest.caseid.nunique())
    gate = "SIGNAL_QC_SCALE_GO" if kept_cases >= 2500 else ("SIGNAL_QC_SCALE_YELLOW" if kept_cases >= 1500 else "SIGNAL_QC_SCALE_NO_GO")
    reasons = qc.loc[~pass_mask,"reject_reason"].value_counts().head(20).to_dict()
    summary = {
        "diagnostic_only": True,
        "diagnostic_name": "P1 60-second end-aligned input sensitivity",
        "canonical_gate0_remains": "20-second preregistered run",
        "input_manifest_windows": int(len(manifest)),
        "qc_pass_windows": int(pass_mask.sum()),
        "qc_pass_fraction": float(pass_mask.mean()),
        "qc_pass_cases": kept_cases,
        "qc_pass_subjects": int(final_manifest.subjectid.nunique()),
        "qc_positive_windows": int(final_manifest.label.sum()),
        "qc_prevalence": float(final_manifest.label.mean()),
        "scale_gate": gate,
        "sampling_rate_hz": FS,
        "window_seconds": WINDOW_SECONDS,
        "end_alignment": "60 s window ends at canonical input_end == prediction_start; prediction horizon remains 300 s",
        "normalization": "same per-window robust normalization/QC as canonical 20 s P1",
        "qc_policy": {
            "min_finite_fraction": MIN_FINITE,
            "max_nan_run_sec": MAX_NAN_RUN_SAMPLES/FS,
            "max_flat_diff_fraction": MAX_FLAT_DIFF,
            "max_robust_outlier_fraction_20sigma": MAX_ROBUST_OUTLIER,
            "ecg_min_p01_p99_range": ECG_MIN_P01_P99_RANGE,
            "pleth_min_p01_p99_range": PLETH_MIN_P01_P99_RANGE,
        },
        "top_reject_reasons": reasons,
        "splits": split_rows,
        "tensor_file": "windows_qc_f16.npy",
        "tensor_shape": [int(pass_mask.sum()), 2, N_SAMPLES],
        "tensor_dtype": "float16",
    }
    (out / "vitaldb_signal_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
