"""Client-update drift metrics for same-shape and nested HeteroFL updates."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


TensorDict = Mapping[str, np.ndarray]


def parameter_delta(after: TensorDict, before: TensorDict) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(after[key], dtype=np.float64) - np.asarray(before[key], dtype=np.float64)
        for key in sorted(set(after) & set(before))
    }


def _norm(delta: TensorDict) -> float:
    return float(np.sqrt(sum(np.sum(value * value) for value in delta.values())))


def _cosine(left: TensorDict, right: TensorDict) -> float:
    dot = left_sq = right_sq = 0.0
    for key in sorted(set(left) & set(right)):
        shape = tuple(min(a, b) for a, b in zip(left[key].shape, right[key].shape))
        slices = tuple(slice(0, size) for size in shape)
        a = left[key][slices].reshape(-1)
        b = right[key][slices].reshape(-1)
        dot += float(a @ b)
        left_sq += float(a @ a)
        right_sq += float(b @ b)
    denominator = np.sqrt(left_sq * right_sq)
    return float(dot / denominator) if denominator else float("nan")


def _pad_weighted_sum(
    deltas: Sequence[TensorDict], weights: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key in sorted(set().union(*(set(delta) for delta in deltas))):
        arrays = [delta[key] for delta in deltas if key in delta]
        maximum = tuple(max(dimensions) for dimensions in zip(*(array.shape for array in arrays)))
        total = np.zeros(maximum, dtype=np.float64)
        for weight, delta in zip(weights, deltas):
            if key not in delta:
                continue
            value = delta[key]
            slices = tuple(slice(0, size) for size in value.shape)
            total[slices] += weight * value
        output[key] = total
    return output


def update_drift_report(
    client_updates: Sequence[TensorDict],
    client_references: Sequence[TensorDict],
    sample_weights: Sequence[float],
    *,
    client_ids: Sequence[int] | None = None,
) -> dict:
    if not client_updates or len(client_updates) != len(client_references):
        raise ValueError("updates and references must have equal non-zero length")
    weights = np.asarray(sample_weights, dtype=np.float64)
    if len(weights) != len(client_updates) or weights.sum() <= 0:
        raise ValueError("sample_weights must match clients and have positive sum")
    weights /= weights.sum()
    ids = list(client_ids or range(len(client_updates)))
    deltas = [
        parameter_delta(update, reference)
        for update, reference in zip(client_updates, client_references)
    ]
    aggregate = _pad_weighted_sum(deltas, weights)
    norms = [_norm(delta) for delta in deltas]
    aggregate_norm = _norm(aggregate)
    denominator = float(np.sum(weights * np.asarray(norms)))
    pairwise = [
        {"left": ids[left], "right": ids[right],
         "cosine": _cosine(deltas[left], deltas[right])}
        for left, right in combinations(range(len(deltas)), 2)
    ]
    return {
        "clients": [
            {
                "client_id": client_id,
                "sample_weight": float(weight),
                "update_norm": norm,
                "cosine_to_global_update": _cosine(delta, aggregate),
            }
            for client_id, weight, norm, delta in zip(ids, weights, norms, deltas)
        ],
        "pairwise_update_cosines": pairwise,
        "global_update_norm": aggregate_norm,
        "aggregation_cancellation_ratio": (
            aggregate_norm / denominator if denominator else float("nan")
        ),
    }
