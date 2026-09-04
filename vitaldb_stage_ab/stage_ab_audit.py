#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://api.vitaldb.net"
TARGETS = {
    "ECG_II": "SNUADC/ECG_II",
    "PLETH": "SNUADC/PLETH",
    "ART_MBP": "Solar8000/ART_MBP",
    "ART": "SNUADC/ART",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch(session: requests.Session, url: str, timeout: int = 180) -> tuple[bytes, dict]:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    meta = {
        "url": url,
        "status_code": r.status_code,
        "content_type": r.headers.get("content-type"),
        "content_encoding": r.headers.get("content-encoding"),
        "content_length_header": r.headers.get("content-length"),
        "received_bytes": len(r.content),
        "etag": r.headers.get("etag"),
        "last_modified": r.headers.get("last-modified"),
    }
    return r.content, meta


def decode_csv(payload: bytes) -> pd.DataFrame:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return pd.read_csv(io.BytesIO(payload))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def case_set(trks: pd.DataFrame, tname: str) -> set[int]:
    return set(pd.to_numeric(trks.loc[trks["tname"].eq(tname), "caseid"], errors="coerce").dropna().astype(int))


def subject_count(cases: pd.DataFrame, ids: set[int]) -> int:
    return int(cases.loc[cases["caseid"].isin(ids), "subjectid"].nunique())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vitaldb_stage_ab")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_index"
    raw.mkdir(exist_ok=True)

    s = requests.Session()
    s.headers.update({"User-Agent": "VitalDB-GoNoGo-audit/0.1"})

    root_bytes, root_meta = fetch(s, API + "/", args.timeout)
    cases_bytes, cases_meta = fetch(s, API + "/cases", args.timeout)
    trks_bytes, trks_meta = fetch(s, API + "/trks", args.timeout)

    (raw / "api_root.html").write_bytes(root_bytes)
    (raw / "cases_response.bin").write_bytes(cases_bytes)
    (raw / "trks_response.bin").write_bytes(trks_bytes)

    cases = decode_csv(cases_bytes)
    trks = decode_csv(trks_bytes)

    required_case_cols = {"caseid", "subjectid"}
    required_trk_cols = {"caseid", "tname", "tid"}
    if not required_case_cols.issubset(cases.columns):
        raise RuntimeError(f"/cases missing columns: {required_case_cols - set(cases.columns)}")
    if not required_trk_cols.issubset(trks.columns):
        raise RuntimeError(f"/trks missing columns: {required_trk_cols - set(trks.columns)}")

    cases["caseid"] = pd.to_numeric(cases["caseid"], errors="raise").astype(int)
    cases["subjectid"] = pd.to_numeric(cases["subjectid"], errors="raise").astype(int)
    trks["caseid"] = pd.to_numeric(trks["caseid"], errors="raise").astype(int)

    cases.to_csv(out / "vitaldb_cases.csv", index=False)
    trks.to_csv(out / "vitaldb_trks.csv", index=False)

    sets = {k: case_set(trks, v) for k, v in TARGETS.items()}
    intersections = {
        "ECG_II": sets["ECG_II"],
        "PLETH": sets["PLETH"],
        "ART_MBP": sets["ART_MBP"],
        "ECG_II+PLETH": sets["ECG_II"] & sets["PLETH"],
        "ECG_II+PLETH+ART_MBP": sets["ECG_II"] & sets["PLETH"] & sets["ART_MBP"],
        "ART+ECG_II": sets["ART"] & sets["ECG_II"],
        "ART+PLETH": sets["ART"] & sets["PLETH"],
    }

    rows = []
    for name, ids in intersections.items():
        rows.append({
            "availability_key": name,
            "n_cases": len(ids),
            "n_subjects": subject_count(cases, ids),
        })
    avail = pd.DataFrame(rows)
    avail.to_csv(out / "vitaldb_track_availability.csv", index=False)

    p1_ids = intersections["ECG_II+PLETH+ART_MBP"]
    p1 = cases.loc[cases["caseid"].isin(p1_ids), ["caseid", "subjectid"]].copy()
    p1["has_ecg"] = True
    p1["has_pleth"] = True
    p1["has_art_mbp"] = True
    p1.sort_values(["subjectid", "caseid"]).to_csv(out / "vitaldb_p1_candidate_cases.csv", index=False)

    n_p1 = len(p1_ids)
    if n_p1 >= 2500:
        raw_scale_gate = "RAW_SCALE_GO"
    elif n_p1 >= 1500:
        raw_scale_gate = "RAW_SCALE_YELLOW"
    else:
        raw_scale_gate = "RAW_SCALE_NO_GO"

    root_text = root_bytes.decode("utf-8", errors="replace")
    root_mentions_ccby4 = "CC BY 4.0" in root_text or "Attribution 4.0" in root_text

    access_audit = {
        "audit_time_utc": now_iso(),
        "route": API,
        "python": sys.version,
        "platform": platform.platform(),
        "endpoints": {
            "root": root_meta,
            "cases": cases_meta,
            "trks": trks_meta,
        },
        "parsed": {
            "n_cases_rows": int(len(cases)),
            "n_unique_caseid": int(cases["caseid"].nunique()),
            "n_unique_subjectid": int(cases["subjectid"].nunique()),
            "n_track_rows": int(len(trks)),
            "n_unique_track_names": int(trks["tname"].nunique()),
            "api_root_mentions_cc_by_4_0": bool(root_mentions_ccby4),
        },
        "license_caution": {
            "status": "UNRESOLVED_WORDING_INCONSISTENCY",
            "note": (
                "The current API landing page identifies the public API as CC BY 4.0, while a legacy official "
                "VitalDB Data Use Agreement page has displayed CC BY-NC-SA 4.0 wording. Do not silently reconcile "
                "these texts. Preserve the exact terms applicable to the chosen data route and consult institutional "
                "policy if required. This script does not make a legal determination."
            ),
        },
    }
    write_json(out / "vitaldb_access_audit.json", access_audit)

    pair_obj = {
        "audit_time_utc": now_iso(),
        "track_names": TARGETS,
        "counts": {k: {"n_cases": len(v), "n_subjects": subject_count(cases, v)} for k, v in intersections.items()},
        "p1_raw_scale_gate": raw_scale_gate,
        "important": "This is pre-signal-QC availability only; it is not the final usable-cohort gate.",
    }
    write_json(out / "vitaldb_pair_availability.json", pair_obj)

    access_md = "# VitalDB access audit\n\n"
    access_md += f"- Audit time (UTC): `{access_audit['audit_time_utc']}`\n"
    access_md += f"- Route: `{API}`\n"
    access_md += f"- `/cases`: HTTP {cases_meta['status_code']}, parsed rows = {len(cases):,}\n"
    access_md += f"- Unique subjects = {cases['subjectid'].nunique():,}\n"
    access_md += f"- `/trks`: HTTP {trks_meta['status_code']}, parsed rows = {len(trks):,}\n"
    access_md += f"- Unique track names = {trks['tname'].nunique():,}\n"
    access_md += "\n## License/DUA caution\n\n"
    access_md += access_audit["license_caution"]["note"] + "\n"
    access_md += "\n## Outcome\n\nAccess and index parsing succeeded. Proceed to availability and smoke signal download.\n"
    (out / "vitaldb_access_audit.md").write_text(access_md, encoding="utf-8")

    md = ["# VitalDB pair availability", "", "Pre-signal-QC counts from the official track index.", "", "| key | cases | subjects |", "|---|---:|---:|"]
    for _, r in avail.iterrows():
        md.append(f"| {r['availability_key']} | {int(r['n_cases']):,} | {int(r['n_subjects']):,} |")
    md += ["", f"P1 raw scale gate: **{raw_scale_gate}**", "", "This is not the final usable-case count; signal-quality and label-construction filters still apply."]
    (out / "vitaldb_pair_availability.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(avail.to_string(index=False))
    print(f"\nP1 raw candidate cases: {n_p1:,} -> {raw_scale_gate}")
    print(f"Artifacts: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
