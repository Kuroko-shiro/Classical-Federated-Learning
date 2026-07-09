"""Client wrapper: binds a model backend to its local data splits.

MIN-lite support: a client may carry an `imputer` = (src_mod, tgt_mod, W, b)
trained pre-FL in a multimodal client (Saha 2024, Sec. 6 — Modality Imputation
Network). When the target modality is missing from this client's data, it is
generated as x[tgt] = x[src] @ W + b at data-access time, turning the incongruent
setting into a pseudo-congruent one. The paper's MIN generates raw radiology
reports with VQ-GAN + a cross-modal transformer; this harness imputes the FEATURE
representation with a linear translator instead. The protocol essence (pre-FL,
trained in one multimodal client, frozen, shipped to unimodal clients, no per-round
overhead) is preserved; the generative machinery is deferred to the torch/IU-X-ray
backend.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.interfaces import FederatedDataset, ModelBackend
from ..core.types import ClientContext


class Client:
    def __init__(self, ctx: ClientContext, model: ModelBackend, dataset: FederatedDataset) -> None:
        self.ctx = ctx
        self.model = model
        self.dataset = dataset
        # (src_mod, tgt_mod, W, b) or None
        self.imputer: Optional[Tuple[str, str, np.ndarray, np.ndarray]] = None

    def _maybe_impute(self, split: Dict[str, Any]) -> Dict[str, Any]:
        if self.imputer is None:
            return split
        src, tgt, W, b = self.imputer
        x = split.get("x", {})
        if tgt not in x and src in x:
            x = dict(x)
            x[tgt] = x[src].astype(np.float32) @ W + b
            split = dict(split)
            split["x"] = x
        return split

    def train_data(self) -> Dict[str, Any]:
        return self._maybe_impute(
            self.dataset.get_split(self.ctx.client_id, "train", self.ctx.modalities))

    def test_data(self) -> Dict[str, Any]:
        # evaluate on the shared test split, restricted to this client's modalities
        return self._maybe_impute(
            self.dataset.get_split(self.ctx.client_id, "test", self.ctx.modalities))
