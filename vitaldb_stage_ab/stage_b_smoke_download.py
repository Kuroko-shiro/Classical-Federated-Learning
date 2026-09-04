#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.vitaldb.net"
TRACKS = ["SNUADC/ECG_II", "SNUADC/PLETH", "Solar8000/ART_MBP"]


def payload_to_csv(payload: bytes) -> pd.DataFrame:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return pd.read_csv(io.BytesIO(payload))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="artifacts/vitaldb_stage_ab")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    base = Path(args.audit_dir)
    cases = pd.read_csv(base / "vitaldb_p1_candidate_cases.csv")
    trks = pd.read_csv(base / "vitaldb_trks.csv")
    out = base / "smoke_tracks"
    out.mkdir(parents=True, exist_ok=True)

    chosen = cases.sort_values(["subjectid", "caseid"]).head(args.n_cases)
    s = requests.Session()
    s.headers.update({"User-Agent": "VitalDB-GoNoGo-smoke/0.1"})

    rows = []
    for caseid in chosen["caseid"].astype(int):
        cdir = out / f"case_{caseid:04d}"
        cdir.mkdir(exist_ok=True)
        c = trks.loc[(trks["caseid"] == caseid) & (trks["tname"].isin(TRACKS))]
        for tname in TRACKS:
            hit = c.loc[c["tname"] == tname]
            if hit.empty:
                rows.append({"caseid": caseid, "tname": tname, "ok": False, "reason": "missing_index"})
                continue
            tid = str(hit.iloc[0]["tid"])
            url = f"{API}/{tid}"
            r = s.get(url, timeout=args.timeout)
            ok = r.status_code == 200
            suffix = ".csv.gz" if r.content[:2] == b"\x1f\x8b" else ".csv"
            fname = tname.replace("/", "__") + suffix
            (cdir / fname).write_bytes(r.content)
            rec = {
                "caseid": caseid,
                "tname": tname,
                "tid": tid,
                "url": url,
                "http_status": r.status_code,
                "download_bytes": len(r.content),
                "ok": ok,
            }
            if ok:
                try:
                    df = payload_to_csv(r.content)
                    rec["n_rows"] = int(len(df))
                    rec["columns"] = list(df.columns)
                    if len(df.columns) >= 2:
                        vals = pd.to_numeric(df.iloc[:, -1], errors="coerce")
                        rec["finite_values"] = int(np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)).sum())
                        rec["value_min"] = float(vals.min()) if vals.notna().any() else None
                        rec["value_max"] = float(vals.max()) if vals.notna().any() else None
                except Exception as e:
                    rec["parse_error"] = repr(e)
            rows.append(rec)
            print(caseid, tname, r.status_code, len(r.content))

    df = pd.DataFrame(rows)
    df.to_csv(base / "vitaldb_smoke_download_audit.csv", index=False)
    (base / "vitaldb_smoke_download_audit.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved smoke audit to {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
