"""Modality-assignment helpers shared by synthetic dataset scenarios."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def assign_modalities(
    num_clients: int,
    modalities: Sequence[str],
    mode: str,
    rng: np.random.Generator,
) -> Dict[int, List[str]]:
    """Assign a non-empty modality subset to each client.

    Supported modes are ``full``, ``disjoint``, ``random`` and ``mixed:k``
    (first *k* clients receive all modalities, the rest receive the first one).
    """

    all_modalities = list(modalities)
    if not all_modalities:
        raise ValueError("at least one modality is required")
    if mode == "full":
        return {cid: list(all_modalities) for cid in range(num_clients)}
    if mode == "disjoint":
        return {cid: [all_modalities[cid % len(all_modalities)]] for cid in range(num_clients)}
    if mode.startswith("mixed:"):
        multimodal = int(mode.split(":", 1)[1])
        if not 0 <= multimodal <= num_clients:
            raise ValueError("mixed:k requires 0 <= k <= num_clients")
        return {
            cid: (list(all_modalities) if cid < multimodal else [all_modalities[0]])
            for cid in range(num_clients)
        }
    if mode == "random":
        out = {}
        for cid in range(num_clients):
            mask = rng.random(len(all_modalities)) < 0.5
            if not mask.any():
                mask[int(rng.integers(0, len(all_modalities)))] = True
            out[cid] = [modality for modality, keep in zip(all_modalities, mask) if keep]
        return out
    raise ValueError(f"unsupported modality mode: {mode}")
