"""LOOT (Leave-One-Out Teacher) — Saha et al. 2024 (arXiv:2402.05294), Sec. 7.2.

Paper: the server fine-tunes each client model (student) so that its feature
embeddings match the mean embeddings of the OTHER K-1 client models (teachers),
using unlabeled public data; this pulls unimodal clients toward multimodal-like
embeddings and was the strongest server-level method in the paper (Table 4).

Harness adaptation (protocol-faithful, placement-adapted): our Strategy protocol
is broadcast -> client_fit -> aggregate, so the leave-one-out alignment runs
inside client_fit on the same public set, against teacher targets computed by the
server from the embeddings every client uploaded in the PREVIOUS round (one-round
lag). The math (match the leave-one-out mean embedding on public data) is the
paper's; only where it executes differs. Embedding upload/download is counted as
communication — LOOT's cost is real and should be measured.

Per round r (for client k):
  broadcast : global params  +  target_k = mean_{j != k} emb_j        [from r-1]
  client_fit: load params -> local CE training -> align_embeddings(public, target_k)
              -> upload new params + emb_k = embed(public)
  aggregate : strip emb keys -> weighted-average params -> store {emb_k} for r+1
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.registry import STRATEGIES
from ..core.types import (
    BroadcastPayload,
    ClientContext,
    FitResult,
    Payload,
    PayloadKind,
)
from ..models.base import weighted_average
from .base import SingleGlobalModelStrategy

_EMB_KEY = "__loot_emb__"


@STRATEGIES.register("loot")
class LOOT(SingleGlobalModelStrategy):
    name = "loot"

    def __init__(self, factory, dataset, public_size: int = 300,
                 align_epochs: int = 1, align_lr: float = 0.05, **kw) -> None:
        super().__init__(factory, dataset, **kw)
        self._public_size = int(public_size)
        self._public = None                               # built in initialize
        self._client_embs: Dict[int, np.ndarray] = {}    # last round's embeddings
        self._align_epochs = int(align_epochs)
        self._align_lr = float(align_lr)

    # global model identical to FedAvg's
    def initialize(self, clients: List[ClientContext]) -> None:
        self._public = self.dataset.public_set(
            self._public_size, self.dataset.modalities,
            np.random.default_rng(0),
        )
        ref = ClientContext(
            client_id=-1, model_name="global",
            modalities=self.dataset.modalities, num_train=0,
            num_classes=self.dataset.num_classes,
            extra=clients[0].extra if clients else {},
        )
        self._global = self.factory.build(ref, self.dataset)

    def broadcast(self, round_idx: int, client: ClientContext) -> BroadcastPayload:
        tensors = dict(self._global.get_parameters(only_shared=True))
        # leave-one-out teacher target from last round's uploads
        others = [e for cid, e in self._client_embs.items() if cid != client.client_id]
        if others:
            tensors[_EMB_KEY] = np.mean(np.stack(others, axis=0), axis=0)
        return Payload(kind=PayloadKind.PARAMETERS, tensors=tensors,
                       meta={"round": round_idx})

    def client_fit(self, model, payload, data, config) -> FitResult:
        params = {k: v for k, v in payload.tensors.items() if k != _EMB_KEY}
        model.set_parameters(params, only_shared=True)
        train_metrics: Dict[str, Any] = {}
        # ORDER MATTERS (matches the paper's cycle): alignment toward last round's
        # leave-one-out teachers FIRST, then local CE training. Local training
        # re-fits the head to the aligned encoders; aligning AFTER training leaves
        # a stale head on shifted embeddings and collapses accuracy.
        if _EMB_KEY in payload.tensors:
            align_loss = model.align_embeddings(
                self._public["x"], payload.tensors[_EMB_KEY],
                epochs=config.get("align_epochs", self._align_epochs),
                lr=config.get("align_lr", self._align_lr),
            )
            train_metrics["loot_align_loss"] = align_loss
        train_metrics.update(model.local_train(
            data, epochs=config.get("local_epochs", 1), lr=config.get("lr", 0.05),
        ))
        update = dict(model.get_parameters(only_shared=True))
        update[_EMB_KEY] = model.embed(self._public["x"])   # upload embeddings
        return FitResult(
            update=Payload(kind=PayloadKind.PARAMETERS, tensors=update),
            num_examples=len(data["y"]),
            train_metrics=train_metrics,
        )

    def aggregate(self, round_idx: int,
                  results: List[Tuple[ClientContext, FitResult]]) -> None:
        params, weights = [], []
        for ctx, fr in results:
            t = dict(fr.update.tensors)
            emb = t.pop(_EMB_KEY, None)
            if emb is not None:
                self._client_embs[ctx.client_id] = emb
            params.append(t)
            weights.append(fr.num_examples)
        new_shared = weighted_average(params, weights)
        self._global.set_parameters(new_shared, only_shared=True)

    def global_model(self):
        return self._global
