#!/usr/bin/env python3
"""Header/availability sanity using vitaldb 1.7.2 without downloading the full cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

TRACKS = ["SNUADC/ECG_II", "SNUADC/PLETH", "Solar8000/ART_MBP"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="artifacts/vitaldb_stage_ab")
    ap.add_argument("--n-cases", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    base = Path(args.audit_dir)
    cand = pd.read_csv(base / "vitaldb_p1_candidate_cases.csv").sort_values(["subjectid", "caseid"]).head(args.n_cases)
    rows = []
    for caseid in cand["caseid"].astype(int):
        rec = {"caseid": caseid}
        try:
            arr = vitaldb.load_case(caseid, TRACKS, interval=args.interval)
            if hasattr(arr, "to_numpy"):
                x = arr.to_numpy()
            else:
                x = np.asarray(arr)
            rec["shape"] = list(x.shape)
            if x.ndim == 2 and x.shape[1] >= 3:
                for j, name in enumerate(["ecg", "pleth", "map"]):
                    v = x[:, j].astype(float)
                    finite = np.isfinite(v)
                    rec[f"{name}_finite_fraction"] = float(finite.mean()) if len(v) else 0.0
                    rec[f"{name}_min"] = float(np.nanmin(v)) if finite.any() else None
                    rec[f"{name}_max"] = float(np.nanmax(v)) if finite.any() else None
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = repr(e)
        rows.append(rec)
        print(rec)

    pd.DataFrame(rows).to_csv(base / "vitaldb_header_qc_smoke.csv", index=False)
    (base / "vitaldb_header_qc_smoke.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
