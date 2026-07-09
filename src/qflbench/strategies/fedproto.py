"""FedProto — prototype-based strategy for MODEL heterogeneity (scenario 2).

Key idea: clients never share parameters (their architectures may differ). Instead
each client computes per-class mean embeddings ("prototypes") and sends those. The
server averages prototypes across clients into global prototypes and broadcasts
them back. Locally, the global prototypes act as a regularizer pulling each class's
embeddings toward the consensus; classification can be done by nearest prototype.

Why this fits the project:
  - works across heterogeneous architectures (only an embedding space is shared);
  - communicates only (num_classes x embed_dim) scalars -> very low comms, which is
    exactly the kind of communication-light protocol the quantum-comms story wants
    to compare against.

This implementation keeps it simple and inspectable:
  - client_fit: local SGD (cross-entropy) + return prototypes. (A proximal pull to
    global prototypes can be added via `extra`; left as a documented TODO so the
    baseline stays transparent.)
  - aggregate: weighted mean of prototypes per class.
  - evaluation: there is no single global *model*; we evaluate per client using
    each client's own encoder + the GLOBAL prototypes via nearest-prototype. The
    engine handles per-client evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.interfaces import ModelBackend, ModelFactory
from ..core.registry import STRATEGIES
from ..core.types import (
    BroadcastPayload,
    ClientContext,
    FitResult,
    Payload,
    PayloadKind,
)
from ..core.interfaces import FederatedStrategy


@STRATEGIES.register("fedproto")
class FedProto(FederatedStrategy):
    name = "fedproto"

    def __init__(self, factory: ModelFactory, dataset, proto_mu: float = 1.0, **kwargs) -> None:
        self.factory = factory
        self.dataset = dataset
        self.proto_mu = float(proto_mu)
        self._global_protos: Dict[int, np.ndarray] = {}
        self._embed_dim: Optional[int] = None

    def initialize(self, clients: List[ClientContext]) -> None:
        self._global_protos = {}

    def broadcast(self, round_idx, client) -> BroadcastPayload:
        tensors = {str(c): v for c, v in self._global_protos.items()}
        return Payload(kind=PayloadKind.PROTOTYPES, tensors=tensors,
                       meta={"round": round_idx})

    def client_fit(self, model, payload, data, config) -> FitResult:
        # Decode global prototypes broadcast from the server and feed them to
        # local training as a regularizer (the core FedProto step).
        global_protos = {int(k): v for k, v in payload.tensors.items()}
        train_metrics = model.local_train(
            data,
            epochs=config.get("local_epochs", 1),
            lr=config.get("lr", 0.05),
            extra={
                "proto_mu": self.proto_mu,
                "global_prototypes": global_protos if global_protos else None,
            },
        )
        protos = model.class_prototypes(data)
        tensors = {str(c): v for c, v in protos.items()}
        # carry per-class counts so the server can weight prototypes properly
        y = data["y"].astype(int)
        counts = {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)}
        return FitResult(
            update=Payload(kind=PayloadKind.PROTOTYPES, tensors=tensors,
                           meta={"counts": counts}),
            num_examples=train_metrics.get("n", len(y)),
            train_metrics=train_metrics,
        )

    def aggregate(self, round_idx, results) -> None:
        # weighted mean per class across clients
        acc: Dict[int, np.ndarray] = {}
        wsum: Dict[int, float] = {}
        for _, fr in results:
            counts = fr.update.meta.get("counts", {})
            for ck, vec in fr.update.tensors.items():
                c = int(ck)
                w = float(counts.get(ck, 1))
                acc[c] = acc.get(c, np.zeros_like(vec)) + w * vec
                wsum[c] = wsum.get(c, 0.0) + w
        self._global_protos = {c: (acc[c] / wsum[c]).astype(np.float32) for c in acc}

    def global_model(self) -> Optional[ModelBackend]:
        return None  # personalized: evaluation is per-client (engine handles it)

    # helper exposed for the engine's per-client evaluation
    def global_prototypes(self) -> Dict[int, np.ndarray]:
        return self._global_protos
