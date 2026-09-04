#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

API = "https://api.vitaldb.net"
ECG = "SNUADC/ECG_II"
PLETH = "SNUADC/PLETH"
MAP = "Solar8000/ART_MBP"

MAP_INTERVAL = 1.0
MAP_MIN, MAP_MAX = 20.0, 250.0
MAX_MAP_GAP = 5.0
MIN_EVENT_SEC = 60.0
MIN_CASE_SPAN = 600.0
MIN_GLOBAL_COVERAGE = 0.80
MIN_LOCAL_COVERAGE = 0.90

OBS_SEC = 80.0
INPUT_OFFSET_SEC = 60.0
INPUT_SEC = 20.0
PRED_SEC = 300.0
SLACK_SEC = 60.0
FRAME_SEC = OBS_SEC + PRED_SEC + SLACK_SEC
GRID_SEC = 60.0
MAX_WINDOWS_PER_CASE = 20


def detect_events(t, v):
    events = []
    start = None
    low = v < 65.0
    for i in range(len(t)):
        if not low[i]:
            if start is not None:
                j = i - 1
                dur = float(t[j] - t[start])
                if dur >= MIN_EVENT_SEC:
                    events.append((float(t[start]), float(t[j]), dur))
                start = None
            continue
        if start is None:
            start = i
        elif i > 0 and t[i] - t[i - 1] > MAX_MAP_GAP:
            j = i - 1
            dur = float(t[j] - t[start])
            if dur >= MIN_EVENT_SEC:
                events.append((float(t[start]), float(t[j]), dur))
            start = i
    if start is not None:
        j = len(t) - 1
        dur = float(t[j] - t[start])
        if dur >= MIN_EVENT_SEC:
            events.append((float(t[start]), float(t[j]), dur))
    return events


def coverage(t, start, end):
    tt = t[(t >= start) & (t <= end)]
    if len(tt) < 2:
        return 0.0
    dt = np.diff(tt)
    return min(1.0, float(np.sum(dt[(dt > 0) & (dt <= MAX_MAP_GAP)])) / max(1.0, end - start))


def overlaps(events, start, end):
    return any(es < end and ee >= start for es, ee, _ in events)


def onset_in(events, start, end):
    return any(start <= es < end for es, _, _ in events)


def load_map(caseid):
    x = vitaldb.load_case(caseid, [MAP], interval=MAP_INTERVAL)
    x = np.asarray(x)
    if x.ndim == 1:
        v = x.astype(float)
    elif x.ndim == 2 and x.shape[1] >= 1:
        v = x[:, 0].astype(float)
    else:
        return np.empty(0), np.empty(0)
    all_t = np.arange(len(v), dtype=float) * MAP_INTERVAL
    finite = np.isfinite(v)
    t, v = all_t[finite], v[finite]
    phys = (v >= MAP_MIN) & (v <= MAP_MAX)
    return t[phys], v[phys]


def process_case(caseid, subjectid, seed):
    st = time.time()
    try:
        t, v = load_map(caseid)
        rec = {
            "caseid": caseid,
            "subjectid": subjectid,
            "map_finite_phys_n": int(len(v)),
            "load_seconds": float(time.time() - st),
        }
        if len(t) < 250:
            rec.update(usable=False, reason="too_few_valid_map")
            return [], rec
        span = float(t[-1] - t[0])
        global_cov = coverage(t, float(t[0]), float(t[-1]))
        rec.update(span_sec=span, global_map_coverage=global_cov)
        if span < MIN_CASE_SPAN or global_cov < MIN_GLOBAL_COVERAGE:
            rec.update(usable=False, reason="map_qc")
            return [], rec

        events = detect_events(t, v)
        rec.update(
            usable=True,
            reason="ok",
            n_events=int(len(events)),
            has_event=bool(events),
            event_total_duration_sec=float(sum(e[2] for e in events)),
        )

        first = math.ceil(float(t[0]) / GRID_SEC) * GRID_SEC
        last = math.floor((float(t[-1]) - FRAME_SEC) / GRID_SEC) * GRID_SEC
        if last < first:
            rec.update(reason="too_short_for_frame")
            return [], rec

        eligible = []
        n_obs = n_slack = n_cov = 0
        a = first
        while a <= last + 1e-9:
            obs_start = a
            obs_end = a + OBS_SEC
            pred_start = obs_end
            pred_end = pred_start + PRED_SEC
            slack_end = pred_end + SLACK_SEC
            cov = coverage(t, obs_start, slack_end)
            if cov < MIN_LOCAL_COVERAGE:
                n_cov += 1
                a += GRID_SEC
                continue
            if overlaps(events, obs_start, obs_end):
                n_obs += 1
                a += GRID_SEC
                continue
            label = 1 if onset_in(events, pred_start, pred_end) else 0
            if label == 0 and overlaps(events, pred_end, slack_end):
                n_slack += 1
                a += GRID_SEC
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
            a += GRID_SEC

        rec.update(
            n_eligible=int(len(eligible)),
            skipped_obs=int(n_obs),
            skipped_slack=int(n_slack),
            skipped_low_map_coverage=int(n_cov),
        )
        if not eligible:
            rec["reason"] = "no_eligible_windows"
            return [], rec

        rng = np.random.default_rng(seed + caseid * 1000003)
        if len(eligible) > MAX_WINDOWS_PER_CASE:
            idx = np.sort(rng.choice(len(eligible), size=MAX_WINDOWS_PER_CASE, replace=False))
            selected = [eligible[i] for i in idx]
        else:
            selected = eligible
        rec.update(
            n_selected=int(len(selected)),
            n_selected_positive=int(sum(w["label"] for w in selected)),
        )
        return selected, rec
    except Exception as e:
        return [], {
            "caseid": caseid,
            "subjectid": subjectid,
            "usable": False,
            "reason": repr(e),
            "load_seconds": float(time.time() - st),
        }


def subject_split(manifest, seed):
    subj = manifest.groupby("subjectid").agg(
        has_positive=("label", "max"),
        n_windows=("label", "size"),
        n_cases=("caseid", "nunique"),
    ).reset_index()
    rng = np.random.default_rng(seed)
    mp = {}
    for cls in [0, 1]:
        ids = subj.loc[subj["has_positive"] == cls, "subjectid"].to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        a = int(round(0.70 * n))
        b = a + int(round(0.15 * n))
        for sid in ids[:a]: mp[int(sid)] = "train"
        for sid in ids[a:b]: mp[int(sid)] = "val"
        for sid in ids[b:]: mp[int(sid)] = "test"
    subj["split"] = subj["subjectid"].map(mp)
    return mp, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vitaldb_stage_f")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(f"{API}/cases")
    trks = pd.read_csv(f"{API}/trks")
    cases["caseid"] = cases["caseid"].astype(int)
    cases["subjectid"] = cases["subjectid"].astype(int)
    trks["caseid"] = trks["caseid"].astype(int)
    def ids(name):
        return set(trks.loc[trks["tname"].eq(name), "caseid"].astype(int))
    p1ids = ids(ECG) & ids(PLETH) & ids(MAP)
    cohort = cases.loc[cases["caseid"].isin(p1ids), ["caseid", "subjectid"]].sort_values(["subjectid", "caseid"])

    rows, audits = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_case, int(r.caseid), int(r.subjectid), args.seed): int(r.caseid)
                for r in cohort.itertuples(index=False)}
        for i, fut in enumerate(as_completed(futs), 1):
            w, audit = fut.result()
            rows.extend(w)
            audits.append(audit)
            if i % 250 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)}", flush=True)

    audit = pd.DataFrame(audits).sort_values("caseid")
    audit.to_csv(out / "vitaldb_packed_map_case_audit.csv", index=False)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No windows generated from packed MAP")
    manifest = manifest.sort_values(["subjectid", "caseid", "input_start"]).reset_index(drop=True)
    mp, subj = subject_split(manifest, args.seed)
    manifest["split"] = manifest["subjectid"].map(mp)
    manifest["window_seq"] = manifest.groupby("caseid").cumcount()
    manifest["window_id"] = manifest.apply(lambda r: f"{int(r.caseid)}_{int(r.window_seq):03d}", axis=1)
    manifest["signal_qc_pending"] = True
    manifest = manifest[[
        "subjectid","caseid","window_id","input_start","input_end",
        "prediction_start","prediction_end","label","split","has_ecg","has_pleth",
        "has_art_mbp","observation_start","slack_end","map_local_coverage","signal_qc_pending"
    ]]
    manifest.to_csv(out / "vitaldb_window_manifest.csv", index=False)
    manifest.to_parquet(out / "vitaldb_window_manifest.parquet", index=False)
    subj.to_csv(out / "vitaldb_subject_split.csv", index=False)

    splits=[]
    for sp in ["train","val","test"]:
        m=manifest[manifest.split==sp]
        splits.append({"split":sp,"n_subjects":int(m.subjectid.nunique()),"n_cases":int(m.caseid.nunique()),
                       "n_windows":int(len(m)),"n_positive":int(m.label.sum()),"prevalence":float(m.label.mean())})
    pd.DataFrame(splits).to_csv(out / "vitaldb_split_summary.csv", index=False)
    usable_cases=int(audit["usable"].fillna(False).sum())
    window_cases=int(manifest.caseid.nunique())
    gate="WINDOW_SCALE_GO" if window_cases>=2500 else ("WINDOW_SCALE_YELLOW" if window_cases>=1500 else "WINDOW_SCALE_NO_GO")
    summary={
        "timebase":"versioned packed .vital 1.0.1; MAP and future ECG/PLETH share the same recording-relative timeline",
        "p1_raw_cases":int(len(cohort)),
        "packed_map_usable_cases":usable_cases,
        "packed_map_usable_subjects":int(audit.loc[audit.usable.fillna(False),"subjectid"].nunique()),
        "cases_with_selected_windows":window_cases,
        "subjects_with_selected_windows":int(manifest.subjectid.nunique()),
        "selected_windows":int(len(manifest)),
        "selected_positive_windows":int(manifest.label.sum()),
        "selected_window_prevalence":float(manifest.label.mean()),
        "max_windows_per_case":MAX_WINDOWS_PER_CASE,
        "grid_sec":GRID_SEC,
        "split_seed":args.seed,
        "window_scale_gate":gate,
        "splits":splits,
        "vitaldb_source_commit":"98389fa74b1d60b9d9a629da9a5ee85dfa81e478",
    }
    (out/"vitaldb_packed_manifest_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__":
    main()
