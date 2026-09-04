#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.vitaldb.net"
ECG = "SNUADC/ECG_II"
PLETH = "SNUADC/PLETH"
MAP = "Solar8000/ART_MBP"

OBS_SEC = 80.0
INPUT_OFFSET_SEC = 60.0
INPUT_SEC = 20.0
PRED_SEC = 300.0
SLACK_SEC = 60.0
TOTAL_FRAME_SEC = OBS_SEC + PRED_SEC + SLACK_SEC
GRID_SEC = 60.0
MAX_WINDOWS_PER_CASE = 20
MAP_MIN = 20.0
MAP_MAX = 250.0
MAX_MAP_GAP_SEC = 5.0
MIN_LOCAL_COVERAGE = 0.90


def fetch_bytes(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "VitalDB-GoNoGo-window-manifest/0.1"})
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


def local_coverage(t: np.ndarray, start: float, end: float) -> float:
    m = (t >= start) & (t <= end)
    tt = t[m]
    if len(tt) < 2:
        return 0.0
    dt = np.diff(tt)
    covered = float(np.sum(dt[(dt > 0) & (dt <= MAX_MAP_GAP_SEC)]))
    return min(1.0, covered / max(1.0, end - start))


def event_overlaps(events, start: float, end: float) -> bool:
    for es, ee, _ in events:
        if es < end and ee >= start:
            return True
    return False


def event_onset_in(events, start: float, end: float) -> bool:
    return any(start <= es < end for es, _, _ in events)


def process_case(caseid: int, subjectid: int, tid: str, seed: int, timeout: int, retries: int):
    try:
        df = read_csv_payload(fetch_bytes(f"{API}/{tid}", timeout, retries))
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(t) & np.isfinite(v)
        t, v = t[finite], v[finite]
        if len(t) < 250:
            return [], {"caseid": caseid, "subjectid": subjectid, "usable": False, "reason": "too_few_finite"}
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        keep = np.r_[t[1:] != t[:-1], True]
        t, v = t[keep], v[keep]
        phys = (v >= MAP_MIN) & (v <= MAP_MAX)
        t, v = t[phys], v[phys]
        if len(t) < 250:
            return [], {"caseid": caseid, "subjectid": subjectid, "usable": False, "reason": "too_few_phys_valid"}
        span = float(t[-1] - t[0])
        dt = np.diff(t)
        global_cov = float(np.sum(dt[(dt > 0) & (dt <= MAX_MAP_GAP_SEC)])) / span if span > 0 else 0.0
        if span < 600.0 or global_cov < 0.80:
            return [], {"caseid": caseid, "subjectid": subjectid, "usable": False, "reason": "label_qc"}

        events = detect_events(t, v)
        first = math.ceil(t[0] / GRID_SEC) * GRID_SEC
        last = math.floor((t[-1] - TOTAL_FRAME_SEC) / GRID_SEC) * GRID_SEC
        if last < first:
            return [], {"caseid": caseid, "subjectid": subjectid, "usable": False, "reason": "too_short_for_frame"}

        eligible = []
        skipped_obs = skipped_slack = skipped_cov = 0
        anchor = first
        while anchor <= last + 1e-9:
            obs_start = anchor
            obs_end = anchor + OBS_SEC
            pred_start = obs_end
            pred_end = pred_start + PRED_SEC
            slack_start = pred_end
            slack_end = slack_start + SLACK_SEC
            cov = local_coverage(t, obs_start, slack_end)
            if cov < MIN_LOCAL_COVERAGE:
                skipped_cov += 1
                anchor += GRID_SEC
                continue
            if event_overlaps(events, obs_start, obs_end):
                skipped_obs += 1
                anchor += GRID_SEC
                continue
            label = 1 if event_onset_in(events, pred_start, pred_end) else 0
            if label == 0 and event_overlaps(events, slack_start, slack_end):
                skipped_slack += 1
                anchor += GRID_SEC
                continue
            eligible.append({
                "subjectid": subjectid,
                "caseid": caseid,
                "observation_start": obs_start,
                "input_start": obs_start + INPUT_OFFSET_SEC,
                "input_end": obs_start + INPUT_OFFSET_SEC + INPUT_SEC,
                "prediction_start": pred_start,
                "prediction_end": pred_end,
                "slack_end": slack_end,
                "label": int(label),
                "map_local_coverage": float(cov),
                "has_ecg": True,
                "has_pleth": True,
                "has_art_mbp": True,
            })
            anchor += GRID_SEC

        if not eligible:
            return [], {"caseid": caseid, "subjectid": subjectid, "usable": True, "reason": "no_eligible_windows", "n_eligible": 0}

        rng = np.random.default_rng(seed + caseid * 1000003)
        if len(eligible) > MAX_WINDOWS_PER_CASE:
            idx = np.sort(rng.choice(len(eligible), size=MAX_WINDOWS_PER_CASE, replace=False))
            selected = [eligible[i] for i in idx]
        else:
            selected = eligible
        return selected, {
            "caseid": caseid,
            "subjectid": subjectid,
            "usable": True,
            "reason": "ok",
            "n_events": len(events),
            "n_eligible": len(eligible),
            "n_selected": len(selected),
            "n_selected_pos": int(sum(x["label"] for x in selected)),
            "skipped_obs": skipped_obs,
            "skipped_slack": skipped_slack,
            "skipped_low_map_coverage": skipped_cov,
            "span_sec": span,
            "global_map_coverage": global_cov,
        }
    except Exception as e:
        return [], {"caseid": caseid, "subjectid": subjectid, "usable": False, "reason": repr(e)}


def stratified_subject_split(manifest: pd.DataFrame, seed: int):
    subj = manifest.groupby("subjectid").agg(
        has_positive=("label", "max"),
        n_windows=("label", "size"),
        n_cases=("caseid", "nunique"),
    ).reset_index()
    rng = np.random.default_rng(seed)
    split_map = {}
    for cls in [0, 1]:
        ids = subj.loc[subj["has_positive"] == cls, "subjectid"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        for sid in ids[:n_train]:
            split_map[int(sid)] = "train"
        for sid in ids[n_train:n_train + n_val]:
            split_map[int(sid)] = "val"
        for sid in ids[n_train + n_val:]:
            split_map[int(sid)] = "test"
    subj["split"] = subj["subjectid"].map(split_map)
    return split_map, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vitaldb_stage_d")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = read_csv_payload(fetch_bytes(API + "/cases", args.timeout, args.retries))
    trks = read_csv_payload(fetch_bytes(API + "/trks", args.timeout, args.retries))
    cases["caseid"] = pd.to_numeric(cases["caseid"], errors="raise").astype(int)
    cases["subjectid"] = pd.to_numeric(cases["subjectid"], errors="raise").astype(int)
    trks["caseid"] = pd.to_numeric(trks["caseid"], errors="raise").astype(int)

    def ids(name):
        return set(trks.loc[trks["tname"].eq(name), "caseid"].astype(int))
    p1_ids = ids(ECG) & ids(PLETH) & ids(MAP)
    cohort = cases.loc[cases["caseid"].isin(p1_ids), ["caseid", "subjectid"]].copy()
    mt = trks.loc[trks["tname"].eq(MAP), ["caseid", "tid"]].drop_duplicates("caseid")
    tid_map = dict(zip(mt["caseid"].astype(int), mt["tid"].astype(str)))

    rows, case_stats = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_case, int(r.caseid), int(r.subjectid), tid_map[int(r.caseid)], args.seed, args.timeout, args.retries): int(r.caseid)
            for r in cohort.itertuples(index=False)
        }
        for i, fut in enumerate(as_completed(futs), start=1):
            w, st = fut.result()
            rows.extend(w)
            case_stats.append(st)
            if i % 250 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)} cases", flush=True)

    manifest = pd.DataFrame(rows)
    stats = pd.DataFrame(case_stats).sort_values("caseid")
    stats.to_csv(out / "vitaldb_window_case_audit.csv", index=False)
    if manifest.empty:
        raise RuntimeError("No eligible windows")

    manifest = manifest.sort_values(["subjectid", "caseid", "input_start"]).reset_index(drop=True)
    split_map, subject_split = stratified_subject_split(manifest, args.seed)
    manifest["split"] = manifest["subjectid"].map(split_map)
    manifest["window_id"] = manifest.groupby("caseid").cumcount().map(lambda x: f"{x:03d}")
    manifest["window_id"] = manifest["caseid"].astype(str) + "_" + manifest["window_id"]
    manifest["signal_qc_pending"] = True

    cols = [
        "subjectid", "caseid", "window_id", "input_start", "input_end",
        "prediction_start", "prediction_end", "label", "split",
        "has_ecg", "has_pleth", "has_art_mbp", "observation_start",
        "slack_end", "map_local_coverage", "signal_qc_pending",
    ]
    manifest = manifest[cols]
    manifest.to_csv(out / "vitaldb_window_manifest.csv", index=False)
    manifest.to_parquet(out / "vitaldb_window_manifest.parquet", index=False)
    subject_split.to_csv(out / "vitaldb_subject_split.csv", index=False)

    split_summary = []
    for split in ["train", "val", "test"]:
        m = manifest[manifest["split"] == split]
        split_summary.append({
            "split": split,
            "n_subjects": int(m["subjectid"].nunique()),
            "n_cases": int(m["caseid"].nunique()),
            "n_windows": int(len(m)),
            "n_positive": int(m["label"].sum()),
            "prevalence": float(m["label"].mean()),
        })
    ssum = pd.DataFrame(split_summary)
    ssum.to_csv(out / "vitaldb_split_summary.csv", index=False)

    n_cases = int(manifest["caseid"].nunique())
    if n_cases >= 2500:
        scale_gate = "WINDOW_SCALE_GO"
    elif n_cases >= 1500:
        scale_gate = "WINDOW_SCALE_YELLOW"
    else:
        scale_gate = "WINDOW_SCALE_NO_GO"
    summary = {
        "p1_raw_cases": int(len(cohort)),
        "map_label_usable_cases": int(stats.get("usable", pd.Series(dtype=bool)).fillna(False).sum()),
        "cases_with_selected_windows": n_cases,
        "subjects_with_selected_windows": int(manifest["subjectid"].nunique()),
        "selected_windows": int(len(manifest)),
        "selected_positive_windows": int(manifest["label"].sum()),
        "selected_window_prevalence": float(manifest["label"].mean()),
        "max_windows_per_case": MAX_WINDOWS_PER_CASE,
        "grid_sec": GRID_SEC,
        "min_local_map_coverage": MIN_LOCAL_COVERAGE,
        "split_seed": args.seed,
        "window_scale_gate": scale_gate,
        "signal_qc_pending": True,
        "splits": split_summary,
    }
    (out / "vitaldb_window_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# VitalDB Stage D — leakage-free window manifest",
        "",
        f"- P1 raw cases: **{summary['p1_raw_cases']:,}**",
        f"- Cases with selected windows: **{n_cases:,}**",
        f"- Subjects: **{summary['subjects_with_selected_windows']:,}**",
        f"- Selected windows: **{summary['selected_windows']:,}**",
        f"- Positive windows: **{summary['selected_positive_windows']:,}**",
        f"- Natural sampled prevalence: **{summary['selected_window_prevalence']:.4f}**",
        f"- Scale gate before ECG/PLETH QC: **{scale_gate}**",
        "",
        "All windows from a subject are assigned to one split. ECG/PLETH signal QC is still pending.",
        "",
        "## Split summary",
        "",
        ssum.to_markdown(index=False),
    ]
    (out / "vitaldb_window_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
