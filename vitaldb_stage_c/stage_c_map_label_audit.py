#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.vitaldb.net"
ECG = "SNUADC/ECG_II"
PLETH = "SNUADC/PLETH"
MAP = "Solar8000/ART_MBP"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_bytes(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "VitalDB-GoNoGo-label-audit/0.1"})
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last!r}")


def read_csv_payload(payload: bytes) -> pd.DataFrame:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return pd.read_csv(io.BytesIO(payload))


def detect_events(t: np.ndarray, v: np.ndarray, threshold: float = 65.0,
                  min_duration: float = 60.0, max_gap: float = 5.0):
    """Detect sustained low-MAP runs on original numeric timestamps.

    Continuity is broken by a MAP value >= threshold or by a gap > max_gap seconds.
    Duration is conservatively measured as last low timestamp - first low timestamp.
    """
    events = []
    if len(t) < 2:
        return events
    low = v < threshold
    start = None
    for i in range(len(t)):
        if not low[i]:
            if start is not None:
                end = i - 1
                dur = float(t[end] - t[start])
                if dur >= min_duration:
                    events.append((float(t[start]), float(t[end]), dur))
                start = None
            continue
        if start is None:
            start = i
        elif i > 0 and (t[i] - t[i - 1] > max_gap):
            end = i - 1
            dur = float(t[end] - t[start])
            if dur >= min_duration:
                events.append((float(t[start]), float(t[end]), dur))
            start = i
    if start is not None:
        end = len(t) - 1
        dur = float(t[end] - t[start])
        if dur >= min_duration:
            events.append((float(t[start]), float(t[end]), dur))
    return events


def audit_one(rec: dict, map_tid: str, timeout: int, retries: int) -> dict:
    caseid = int(rec["caseid"])
    subjectid = int(rec["subjectid"])
    out = {"caseid": caseid, "subjectid": subjectid, "map_tid": map_tid}
    try:
        payload = fetch_bytes(f"{API}/{map_tid}", timeout=timeout, retries=retries)
        df = read_csv_payload(payload)
        if df.shape[1] < 2:
            raise ValueError(f"unexpected MAP payload columns: {list(df.columns)}")
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(t) & np.isfinite(v)
        out["download_bytes"] = int(len(payload))
        out["n_rows"] = int(len(df))
        out["n_finite"] = int(finite.sum())
        if finite.sum() < 2:
            out.update({"ok": False, "label_usable": False, "reason": "too_few_finite"})
            return out

        t = t[finite]
        v = v[finite]
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        # Collapse duplicate timestamps by keeping the last measurement.
        keep = np.r_[t[1:] != t[:-1], True]
        t, v = t[keep], v[keep]

        # Broad physiological guardrail for numeric MAP label construction.
        phys = (v >= 20.0) & (v <= 250.0)
        out["phys_valid_fraction"] = float(phys.mean()) if len(v) else 0.0
        out["raw_min"] = float(np.nanmin(v))
        out["raw_max"] = float(np.nanmax(v))
        t2, v2 = t[phys], v[phys]
        out["n_phys_valid"] = int(len(t2))
        if len(t2) < 2:
            out.update({"ok": False, "label_usable": False, "reason": "too_few_phys_valid"})
            return out

        dt = np.diff(t2)
        pos_dt = dt[dt > 0]
        med_dt = float(np.median(pos_dt)) if len(pos_dt) else math.nan
        span = float(t2[-1] - t2[0])
        covered = float(np.sum(dt[(dt > 0) & (dt <= 5.0)]))
        coverage = covered / span if span > 0 else 0.0
        events = detect_events(t2, v2, threshold=65.0, min_duration=60.0, max_gap=5.0)

        out.update({
            "ok": True,
            "time_start": float(t2[0]),
            "time_end": float(t2[-1]),
            "span_sec": span,
            "median_dt_sec": med_dt,
            "continuity_coverage_fraction": coverage,
            "map_lt65_sample_fraction": float(np.mean(v2 < 65.0)),
            "n_events": int(len(events)),
            "has_event": bool(events),
            "first_event_onset": events[0][0] if events else None,
            "event_total_duration_sec": float(sum(e[2] for e in events)),
            # Label-only usability gate. Final usable cohort additionally requires ECG/PLETH QC.
            "label_usable": bool(span >= 600.0 and coverage >= 0.80 and len(t2) >= 250),
            "reason": "ok",
        })
        return out
    except Exception as e:
        out.update({"ok": False, "label_usable": False, "reason": repr(e)})
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vitaldb_stage_c")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all P1 cases")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = read_csv_payload(fetch_bytes(API + "/cases", args.timeout, args.retries))
    trks = read_csv_payload(fetch_bytes(API + "/trks", args.timeout, args.retries))
    cases["caseid"] = pd.to_numeric(cases["caseid"], errors="raise").astype(int)
    cases["subjectid"] = pd.to_numeric(cases["subjectid"], errors="raise").astype(int)
    trks["caseid"] = pd.to_numeric(trks["caseid"], errors="raise").astype(int)

    def ids(name: str) -> set[int]:
        return set(trks.loc[trks["tname"].eq(name), "caseid"].astype(int))

    p1_ids = ids(ECG) & ids(PLETH) & ids(MAP)
    p1 = cases.loc[cases["caseid"].isin(p1_ids), ["caseid", "subjectid"]].sort_values(["subjectid", "caseid"])
    if args.limit > 0:
        p1 = p1.head(args.limit)

    map_rows = trks.loc[trks["tname"].eq(MAP), ["caseid", "tid"]].drop_duplicates("caseid")
    map_tid = dict(zip(map_rows["caseid"].astype(int), map_rows["tid"].astype(str)))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(audit_one, row._asdict(), map_tid[int(row.caseid)], args.timeout, args.retries): int(row.caseid)
            for row in p1.itertuples(index=False)
        }
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 250 == 0 or done == len(futs):
                print(f"completed {done}/{len(futs)}", flush=True)

    audit = pd.DataFrame(results).sort_values("caseid")
    audit.to_csv(outdir / "vitaldb_map_label_audit.csv", index=False)

    usable = audit[audit["label_usable"].fillna(False)]
    summary = {
        "audit_time_utc": now_iso(),
        "p1_raw_cases": int(len(p1)),
        "p1_raw_subjects": int(p1["subjectid"].nunique()),
        "map_download_success_cases": int(audit["ok"].fillna(False).sum()),
        "label_usable_cases": int(len(usable)),
        "label_usable_subjects": int(usable["subjectid"].nunique()) if len(usable) else 0,
        "label_usable_fraction": float(len(usable) / len(p1)) if len(p1) else 0.0,
        "cases_with_qualifying_ioh": int(usable["has_event"].fillna(False).sum()) if len(usable) else 0,
        "case_level_ioh_prevalence": float(usable["has_event"].fillna(False).mean()) if len(usable) else None,
        "total_qualifying_events": int(usable["n_events"].fillna(0).sum()) if len(usable) else 0,
        "median_events_per_positive_case": float(usable.loc[usable["has_event"] == True, "n_events"].median()) if (len(usable) and usable["has_event"].fillna(False).any()) else None,
        "median_map_sampling_interval_sec": float(usable["median_dt_sec"].median()) if len(usable) else None,
        "median_continuity_coverage_fraction": float(usable["continuity_coverage_fraction"].median()) if len(usable) else None,
        "event_definition": "Solar8000/ART_MBP < 65 mmHg continuously for >=60 s; gaps >5 s break continuity; values outside [20,250] mmHg are invalid for label audit.",
        "important": "This is label-only usability. Final usable cohort must also pass ECG/PLETH signal QC and window construction.",
    }
    (outdir / "vitaldb_cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    spec = f"""# VitalDB labeling specification — audit freeze v0.1\n\n## Label source\n\n- Source: `Solar8000/ART_MBP` only.\n- ART waveform is not a P1 model input.\n- Hypotension event: MAP < 65 mmHg sustained for at least 60 seconds.\n- MAP values outside [20, 250] mmHg are treated as invalid for this audit.\n- A gap > 5 seconds between valid MAP measurements breaks event continuity.\n- Event duration is conservatively measured from first to last below-threshold timestamp.\n\n## Prediction framing for Gate 0\n\nAdopt the recent cross-center VitalDB framing as the initial frozen policy:\n\n- Observation window: 80 seconds.\n- First 60 seconds: hypotension-free baseline check.\n- Final 20 seconds: ECG/PLETH model input segment.\n- Prediction window: following 5 minutes.\n- Slack window: following 1 minute.\n- Positive: a qualifying hypotension event begins in the prediction window.\n- Skip: a qualifying event is active in the observation window.\n- Negative: no qualifying event occurs in observation, prediction, or slack windows.\n- Skip rather than label negative when an event begins in the slack window.\n\n## Sampling\n\n- Do not enumerate every overlapping sample for training.\n- Window-sampling seed will be saved.\n- Minimum separation between retained windows from the same case: 60 seconds.\n- Cap retained windows per case before Gate 0 training; exact cap will be selected after the natural candidate-window prevalence audit.\n- Final test evaluation retains natural class prevalence; balancing/weighting is training-only.\n\n## Status\n\nThis file freezes the event and temporal framing. Exact per-case window cap is intentionally deferred until candidate-window prevalence is measured; models must not be compared across different framing rules.\n"""
    (outdir / "vitaldb_labeling_spec.md").write_text(spec, encoding="utf-8")

    md = [
        "# VitalDB Stage C — MAP label audit",
        "",
        f"- Raw P1 cases: **{summary['p1_raw_cases']:,}**",
        f"- Raw P1 subjects: **{summary['p1_raw_subjects']:,}**",
        f"- MAP downloads successful: **{summary['map_download_success_cases']:,}**",
        f"- Label-usable cases: **{summary['label_usable_cases']:,}**",
        f"- Label-usable subjects: **{summary['label_usable_subjects']:,}**",
        f"- Label-usable fraction: **{summary['label_usable_fraction']:.3f}**",
        f"- Cases with qualifying IOH: **{summary['cases_with_qualifying_ioh']:,}**",
        f"- Case-level IOH prevalence: **{summary['case_level_ioh_prevalence']:.3f}**" if summary['case_level_ioh_prevalence'] is not None else "- Case-level IOH prevalence: n/a",
        f"- Total qualifying events: **{summary['total_qualifying_events']:,}**",
        f"- Median MAP interval: **{summary['median_map_sampling_interval_sec']:.3f} s**" if summary['median_map_sampling_interval_sec'] is not None else "- Median MAP interval: n/a",
        "",
        "This is not yet the final usable P1 cohort. ECG/PLETH signal QC and leakage-free window construction remain required.",
    ]
    (outdir / "vitaldb_stage_c_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
