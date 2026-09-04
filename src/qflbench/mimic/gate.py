from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

CONDITIONS = ("ehr_only", "cxr_only", "multimodal")


def evaluate_complementarity_gate(metrics: pd.DataFrame | str | Path, output_dir: str | Path, *, leakage_pass: bool = True, min_seeds_for_go: int = 3) -> dict[str, Any]:
    if not isinstance(metrics, pd.DataFrame):
        metrics = pd.read_csv(metrics)
    df = metrics.copy()
    required = {"condition", "seed", "auprc", "auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics missing required columns: {sorted(missing)}")
    wide = df.pivot_table(index="seed", columns="condition", values="auprc", aggfunc="mean")
    eligible = wide.dropna(subset=list(CONDITIONS)).copy()
    eligible["best_unimodal"] = eligible[["ehr_only", "cxr_only"]].max(axis=1)
    eligible["fusion_gain"] = eligible["multimodal"] - eligible["best_unimodal"]
    gains = eligible["fusion_gain"]
    n = int(len(gains)); mean_gain = float(gains.mean()) if n else None
    all_positive = bool(n and (gains > 0).all())
    if leakage_pass and n >= min_seeds_for_go and all_positive and mean_gain is not None and mean_gain > 0:
        decision = "GO"
    elif mean_gain is not None and mean_gain > 0 and leakage_pass:
        decision = "YELLOW"
    else:
        decision = "NO-GO"
    result = {"decision": decision, "primary_metric": "AUPRC", "n_eligible_seeds": n, "fusion_gain_mean": mean_gain, "fusion_gain_by_seed": {str(k): float(v) for k, v in gains.items()}, "fusion_gain_positive_all_seeds": all_positive, "leakage_pass": bool(leakage_pass)}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "mimic_complementarity_metrics.csv", index=False)
    (out / "mimic_complementarity_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (out / "mimic_complementarity_gate.md").write_text(f"# MIMIC Complementarity Gate 0\n\n**Decision: {decision}**\n\nMean fusion gain: {mean_gain}\n", encoding="utf-8")
    return result
