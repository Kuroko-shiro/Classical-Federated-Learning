"""Model-layer shared helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def filter_keys(params: Dict[str, np.ndarray], keys: List[str]) -> Dict[str, np.ndarray]:
    return {k: v for k, v in params.items() if k in keys}


def weighted_average(
    param_list: List[Dict[str, np.ndarray]], weights: List[float]
) -> Dict[str, np.ndarray]:
    """Weighted elementwise average over a list of compatible TensorDicts.

    Only keys present in ALL dicts are averaged (so heterogeneous models can still
    share their common keys, e.g. fusion+head). This is what lets parameter
    aggregation degrade gracefully when only some parameters are shared.
    """
    if not param_list:
        return {}
    common = set(param_list[0].keys())
    for p in param_list[1:]:
        common &= set(p.keys())
    common_keys = sorted(common)

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out: Dict[str, np.ndarray] = {}
    for k in common_keys:
        stacked = np.stack([p[k].astype(np.float64) for p in param_list], axis=0)
        out[k] = np.tensordot(w, stacked, axes=(0, 0)).astype(np.float32)
    return out


def hetero_aggregate(
    param_list: List[Dict[str, np.ndarray]], weights: List[float]
) -> Dict[str, np.ndarray]:
    """HeteroFL-style nested aggregation for width-varying models (scenario 2).

    Same-shape keys (backbones — identical across clients) -> plain weighted
    mean, byte-for-byte the weighted_average() behaviour.
    Shape-varying keys (img.proj / txt.proj / head under vary_embed) -> each
    client's tensor is the LEADING slice of the max-shape global tensor; every
    entry is averaged over the clients that actually cover it (coverage-
    weighted), so the outer ring is never zero-diluted.
    Returns max-shape global params; hand each client its cut via slice_to().
    """
    if not param_list:
        return {}
    keys = sorted(param_list[0].keys())
    w = np.asarray(weights, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        arrs = [p[k] for p in param_list]
        shapes = {a.shape for a in arrs}
        if len(shapes) == 1:
            stacked = np.stack([a.astype(np.float64) for a in arrs], axis=0)
            out[k] = np.tensordot(w / w.sum(), stacked,
                                  axes=(0, 0)).astype(np.float32)
        else:
            mx = tuple(max(dims) for dims in zip(*(a.shape for a in arrs)))
            acc = np.zeros(mx, dtype=np.float64)
            cov = np.zeros(mx, dtype=np.float64)
            for wi, a in zip(w, arrs):
                sl = tuple(slice(0, n) for n in a.shape)
                acc[sl] += wi * a.astype(np.float64)
                cov[sl] += wi
            out[k] = (acc / np.maximum(cov, 1e-12)).astype(np.float32)
    return out


def slice_to(global_params: Dict[str, np.ndarray],
             ref_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Cut max-shape global params down to one client's shapes (leading
    slices). Keys whose shapes already match (or are absent from ref) pass
    through unchanged."""
    out: Dict[str, np.ndarray] = {}
    for k, v in global_params.items():
        r = ref_params.get(k)
        if r is not None and r.shape != v.shape:
            sl = tuple(slice(0, n) for n in r.shape)
            out[k] = v[sl]
        else:
            out[k] = v
    return out
