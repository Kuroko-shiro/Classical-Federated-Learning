from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED = {"subject_id", "split", "mortality", "has_ehr", "has_cxr"}


def audit_cohort(manifest: pd.DataFrame | str | Path, output_dir: str | Path, *, stay_col="stay_id", study_col="study_id", image_col="dicom_id", cxr_time_col="cxr_time", prediction_time_col="prediction_time") -> dict[str, Any]:
    """Audit the canonical MIMIC mortality manifest without inventing cohort rules."""
    if not isinstance(manifest, pd.DataFrame):
        manifest = pd.read_csv(manifest)
    df = manifest.copy()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df["mortality"] = pd.to_numeric(df["mortality"], errors="raise").astype(int)
    if not set(df["mortality"].dropna().unique()).issubset({0, 1}):
        raise ValueError("mortality must be binary {0,1}")

    split_subjects = {str(s): set(g["subject_id"].astype(str)) for s, g in df.groupby("split", dropna=False)}
    overlaps = []
    names = sorted(split_subjects)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(split_subjects[a] & split_subjects[b])
            if n:
                overlaps.append({"split_a": a, "split_b": b, "n_subjects": n})

    timing_violations = None
    if cxr_time_col in df.columns and prediction_time_col in df.columns:
        c = pd.to_datetime(df[cxr_time_col], errors="coerce")
        p = pd.to_datetime(df[prediction_time_col], errors="coerce")
        timing_violations = int(((c > p) & c.notna() & p.notna()).sum())

    rows = []
    for split, g in df.groupby("split", dropna=False):
        paired = g[g["has_ehr"].astype(bool) & g["has_cxr"].astype(bool)]
        rows.append({
            "split": split,
            "n_samples": int(len(g)),
            "n_unique_patients": int(g["subject_id"].nunique()),
            "n_unique_stays": int(g[stay_col].nunique()) if stay_col in g else None,
            "n_unique_cxr_studies": int(g[study_col].nunique()) if study_col in g else None,
            "n_unique_images": int(g[image_col].nunique()) if image_col in g else None,
            "mortality_positive": int(g["mortality"].sum()),
            "mortality_negative": int((1 - g["mortality"]).sum()),
            "mortality_prevalence": float(g["mortality"].mean()) if len(g) else None,
            "n_fully_paired": int(len(paired)),
            "paired_prevalence": float(paired["mortality"].mean()) if len(paired) else None,
        })
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out / "mimic_mortality_cohort_audit.csv", index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_unique_patients": int(df["subject_id"].nunique()),
        "n_fully_paired": int((df["has_ehr"].astype(bool) & df["has_cxr"].astype(bool)).sum()),
        "patient_split_overlap": overlaps,
        "patient_level_split_separation_pass": len(overlaps) == 0,
        "cxr_after_prediction_time_violations": timing_violations,
        "cxr_timing_pass": timing_violations in (None, 0),
        "mortality_positive": int(df["mortality"].sum()),
        "mortality_negative": int((1 - df["mortality"]).sum()),
        "mortality_prevalence": float(df["mortality"].mean()) if len(df) else None,
        "splits": audit_df.to_dict(orient="records"),
    }
    summary["leakage_audit_pass"] = bool(summary["patient_level_split_separation_pass"] and summary["cxr_timing_pass"])
    (out / "mimic_mortality_cohort_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    notes = ["# MIMIC Mortality Linkage Notes", "", "This report audits the benchmark manifest; it does not invent a cohort definition.", f"- Patient-level split separation: **{summary['patient_level_split_separation_pass']}**", f"- CXR timing check: **{summary['cxr_timing_pass']}**", f"- Overall leakage audit: **{summary['leakage_audit_pass']}**", f"- Fully paired rows: **{summary['n_fully_paired']}**"]
    (out / "mimic_mortality_linkage_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary
