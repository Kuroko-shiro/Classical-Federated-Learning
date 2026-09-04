from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _assign_modalities(k: int, complete_ratio: float, seed: int) -> list[str]:
    n_complete = int(round(k * complete_ratio))
    rem = k - n_complete
    n_ehr = rem // 2 + (rem % 2 if seed % 2 == 0 else 0)
    n_cxr = rem - n_ehr
    labels = ["ehr+cxr"] * n_complete + ["ehr_only"] * n_ehr + ["cxr_only"] * n_cxr
    rng = np.random.default_rng(seed); rng.shuffle(labels)
    return labels


def build_fl_partition(manifest: pd.DataFrame | str | Path, output_dir: str | Path, *, k: int = 10, seed: int = 0, alpha: float | None = None, complete_ratio: float = 0.50, min_positive: int = 1, min_negative: int = 1, max_attempts: int = 1000) -> dict[str, Any]:
    if not isinstance(manifest, pd.DataFrame):
        manifest = pd.read_csv(manifest)
    df = manifest.copy()
    required = {"subject_id", "mortality", "has_ehr", "has_cxr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower().eq("train")].copy()
    df = df[df["has_ehr"].astype(bool) & df["has_cxr"].astype(bool)].copy()
    if df.empty:
        raise ValueError("no fully paired training rows")
    patient_y = df.groupby("subject_id")["mortality"].agg(lambda x: int(round(float(x.max()))))
    patients = patient_y.index.to_numpy(); y = patient_y.to_numpy(dtype=int)
    rng = np.random.default_rng(seed); assignment = None
    for _ in range(max_attempts):
        client_lists = [[] for _ in range(k)]
        if alpha is None or np.isinf(alpha):
            for cls in (0, 1):
                ids = patients[y == cls].copy(); rng.shuffle(ids)
                for j, pid in enumerate(ids): client_lists[j % k].append(pid)
        else:
            if alpha <= 0: raise ValueError("alpha must be positive, inf, or None")
            for cls in (0, 1):
                ids = patients[y == cls].copy(); rng.shuffle(ids)
                props = rng.dirichlet(np.full(k, alpha, dtype=float))
                chunks = np.split(ids, (np.cumsum(props)[:-1] * len(ids)).astype(int))
                for c, chunk in enumerate(chunks): client_lists[c].extend(chunk.tolist())
        ok = True
        for ids in client_lists:
            yy = patient_y.loc[ids] if ids else pd.Series(dtype=int)
            pos = int(yy.sum()); neg = int(len(yy) - pos)
            ok &= pos >= min_positive and neg >= min_negative
        if ok:
            assignment = client_lists; break
    if assignment is None:
        raise RuntimeError("unable to construct viable partition; relax support constraints or alpha")
    modalities = _assign_modalities(k, complete_ratio, seed)
    client_map = {str(pid): c for c, ids in enumerate(assignment) for pid in ids}
    part = df.copy(); part["client_id"] = part["subject_id"].astype(str).map(client_map).astype(int)
    part["client_modality"] = part["client_id"].map(dict(enumerate(modalities)))
    rows = []
    for c, g in part.groupby("client_id"):
        py = g.groupby("subject_id")["mortality"].max()
        rows.append({"client_id": int(c), "modality": modalities[int(c)], "n_rows": int(len(g)), "n_patients": int(len(py)), "positive": int(py.sum()), "negative": int(len(py) - py.sum()), "positive_rate": float(py.mean())})
    stats = pd.DataFrame(rows).sort_values("client_id")
    counts = stats["n_patients"].to_numpy(dtype=float)
    summary = {"k": k, "seed": seed, "alpha": None if alpha is None or np.isinf(alpha) else float(alpha), "partition_mode": "near_iid" if alpha is None or np.isinf(alpha) else "dirichlet_label_skew", "complete_ratio_requested": float(complete_ratio), "modality_counts": stats["modality"].value_counts().to_dict(), "quantity_cv": float(counts.std(ddof=0) / counts.mean()) if counts.mean() else 0.0, "min_positive": int(stats["positive"].min()), "min_negative": int(stats["negative"].min()), "patient_cross_client_overlap": False}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    part.to_csv(out / f"mimic_fl_partition_seed{seed}.csv", index=False)
    stats.to_csv(out / f"mimic_fl_client_stats_seed{seed}.csv", index=False)
    (out / f"mimic_fl_partition_summary_seed{seed}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": part, "client_stats": stats, "summary": summary}
