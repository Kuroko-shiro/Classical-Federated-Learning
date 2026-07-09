"""Consistent analysis metrics across scenarios 1-4.

These are PURE functions over a run summary (the dict returned by RunLogger /
run_experiment), so they apply identically to every scenario and can be re-run on
saved history.json files. This is what makes the four scenarios comparable on one
ruler, and what later lets a QFL run be dropped in for head-to-head comparison.

Three families:
  1. convergence efficiency  -> rounds & cumulative comms to reach a target metric
  2. heterogeneity tax        -> gap vs centralized/local reference lines
  3. modality breakdown        -> accuracy by #modalities a client holds (3/4 only;
                                  degenerate but well-defined for 1/2)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# 1. Convergence efficiency
# --------------------------------------------------------------------------- #
def _round_metric_value(round_rec: Dict[str, Any], metric: str) -> Optional[float]:
    """Pull a scalar metric from a round record, preferring the global model and
    falling back to the per-client mean (so prototype/distillation strategies with
    no single global model still yield a convergence curve)."""
    g = round_rec.get("global_metrics")
    if g and metric in g and g[metric] is not None:
        return float(g[metric])
    extra = round_rec.get("extra") or {}
    mean_key = f"mean_{metric}"
    if mean_key in extra and extra[mean_key] is not None:
        return float(extra[mean_key])
    return None


def rounds_to_target(
    summary: Dict[str, Any], metric: str = "accuracy", target: float = 0.8
) -> Optional[int]:
    """First round index (1-based count) at which `metric` >= target. None if never.

    The single most important efficiency number for the QFL comparison: it answers
    'how many communication rounds did it take to get good enough', independent of
    final accuracy.
    """
    for rec in summary["history"]:
        val = _round_metric_value(rec, metric)
        if val is not None and val >= target:
            return int(rec["round_idx"]) + 1
    return None


def comms_to_target(
    summary: Dict[str, Any], metric: str = "accuracy", target: float = 0.8
) -> Optional[Dict[str, int]]:
    """Cumulative uplink/downlink bytes spent up to and including the round that
    first reaches `target`. None if target never reached.

    This is the 'communication budget to reach good-enough' — the axis on which a
    quantum channel is expected to win or lose. Reported separately from raw total
    comms because a method can be cheap-per-round yet slow to converge."""
    up = down = 0
    for rec in summary["history"]:
        up += int(rec.get("uplink_bytes", 0))
        down += int(rec.get("downlink_bytes", 0))
        val = _round_metric_value(rec, metric)
        if val is not None and val >= target:
            return {"uplink_bytes": up, "downlink_bytes": down,
                    "total_bytes": up + down}
    return None


def area_under_curve(
    summary: Dict[str, Any], metric: str = "accuracy"
) -> Optional[float]:
    """Mean of the per-round metric over all rounds (a simple AUC-of-learning-curve
    proxy). Higher = reaches good performance sooner AND holds it. Robust summary of
    'how good was the whole trajectory', not just the endpoint."""
    vals = [
        _round_metric_value(rec, metric)
        for rec in summary["history"]
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# --------------------------------------------------------------------------- #
# 2. Heterogeneity tax (relative to reference lines)
# --------------------------------------------------------------------------- #
def final_metric(summary: Dict[str, Any], metric: str = "accuracy") -> Optional[float]:
    if not summary["history"]:
        return None
    return _round_metric_value(summary["history"][-1], metric)


def gap_filled_fraction(
    summary: Dict[str, Any], metric: str = "accuracy",
    min_gap: float = 0.05,
) -> Optional[float]:
    """(FL - Local) / (Centralized - Local): fraction of the achievable gap that FL
    closed. Requires baselines in the summary.

    DENOMINATOR GUARD: this metric is only meaningful when there is a real gap
    between the centralized ceiling and the local-only floor. On easy/synthetic
    tasks the two collapse together, making the denominator tiny (or even negative
    when local > centralized due to seed noise), which blows the ratio up to
    nonsensical values (e.g. -14, +3). When |ceiling - floor| < min_gap, we return
    None ("undefined here") rather than a garbage number. On a harder real dataset
    (IU X-ray) the gap is expected to be large and this metric becomes usable again
    automatically — so we keep the concept, but silence it when it cannot be
    computed honestly."""
    base = summary.get("baselines")
    if not base:
        return None
    fl = final_metric(summary, metric)
    cen = base.get("centralized", {}).get(metric)
    loc = base.get("local_only", {}).get(f"mean_{metric}")
    if None in (fl, cen, loc):
        return None
    denom = cen - loc
    # require a real, positive ceiling-floor gap; otherwise the ratio is meaningless
    if denom < min_gap:
        return None
    return float((fl - loc) / denom)


# --------------------------------------------------------------------------- #
# 3. Modality breakdown (meaningful for scenarios 3/4)
# --------------------------------------------------------------------------- #
def modality_breakdown(
    summary: Dict[str, Any],
    client_modalities: Dict[int, List[str]],
    metric: str = "accuracy",
) -> Dict[str, float]:
    """Average final per-client metric grouped by how many modalities each client
    holds. For scenarios 1/2 every client is full-modality so this returns a single
    bucket (degenerate but well-defined); for 3/4 it reveals whether single-modality
    clients are the ones dragging performance down — i.e. WHERE the hole is."""
    if not summary["history"]:
        return {}
    last = summary["history"][-1]
    pcm = last.get("per_client_metrics", {})
    buckets: Dict[int, List[float]] = {}
    for cid_str, m in pcm.items():
        cid = int(cid_str)
        n_mod = len(client_modalities.get(cid, []))
        v = m.get(metric)
        if v is not None:
            buckets.setdefault(n_mod, []).append(float(v))
    return {
        f"{k}_modality": float(sum(vs) / len(vs))
        for k, vs in sorted(buckets.items()) if vs
    }


# --------------------------------------------------------------------------- #
# Convenience: one-shot report row for a run (used to build comparison tables)
# --------------------------------------------------------------------------- #
def summarize_run(
    summary: Dict[str, Any],
    metric: str = "accuracy",
    target: float = 0.8,
    client_modalities: Optional[Dict[int, List[str]]] = None,
) -> Dict[str, Any]:
    """Collapse a run into a single comparable row. Same fields for every scenario
    and (later) for QFL runs, so rows stack directly into a comparison table."""
    last = summary["history"][-1] if summary["history"] else {}
    extra = last.get("extra") or {}
    row = {
        "final_" + metric: final_metric(summary, metric),
        "final_macro_f1": (last.get("global_metrics") or {}).get("macro_f1")
                          or extra.get("mean_macro_f1"),
        "per_client_worst": extra.get(f"worst_{metric}"),
        "per_client_std": extra.get(f"std_{metric}"),
        "rounds_to_target": rounds_to_target(summary, metric, target),
        "comms_to_target_bytes": (comms_to_target(summary, metric, target) or {}).get("total_bytes"),
        "auc_learning": area_under_curve(summary, metric),
        "gap_filled": gap_filled_fraction(summary, metric),
        "total_bytes": summary.get("totals", {}).get("total_uplink_bytes", 0)
                       + summary.get("totals", {}).get("total_downlink_bytes", 0),
        "rounds": summary.get("totals", {}).get("rounds"),
        "target": target,
    }
    if client_modalities is not None:
        row["modality_breakdown"] = modality_breakdown(summary, client_modalities, metric)
    return row
