"""FedAvg and FedProx — parameter-aggregation strategies (scenario 1).

FedAvg: broadcast global shared params -> clients train -> weighted-average the
shared params back. With a homogeneous model the "shared params" can be the whole
model; here we aggregate the head (and any commonly-named params), which also makes
these strategies *work as a baseline* in the modality-subset setting (scenario 3),
where only the head is common.

FedProx: identical except a proximal term mu/2 * ||w - w_global||^2 is added to the
client objective on the shared params. Implemented by threading `proximal_mu` into
the model's local_train — no separate client class needed.
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


@STRATEGIES.register("fedavg")
class FedAvg(SingleGlobalModelStrategy):
    name = "fedavg"
    proximal_mu = 0.0  # overridden by FedProx

    def initialize(self, clients: List[ClientContext]) -> None:
        # Build the global model from a "full-modality" context so it owns a head
        # in the shared embedding space. Encoders here are nominal; only shared
        # keys are ever broadcast/aggregated.
        ref = ClientContext(
            client_id=-1,
            model_name="global",
            modalities=self.dataset.modalities,
            num_train=0,
            num_classes=self.dataset.num_classes,
            extra=clients[0].extra if clients else {},
        )
        self._global = self.factory.build(ref, self.dataset)

    def broadcast(self, round_idx: int, client: ClientContext) -> BroadcastPayload:
        shared = self._global.get_parameters(only_shared=True)
        return Payload(kind=PayloadKind.PARAMETERS, tensors=shared,
                       meta={"round": round_idx})

    def client_fit(self, model, payload, data, config) -> FitResult:
        # load shared params from server, keep local encoders
        model.set_parameters(payload.tensors, only_shared=True)
        train_metrics = model.local_train(
            data,
            epochs=config.get("local_epochs", 1),
            lr=config.get("lr", 0.05),
            proximal_mu=self.proximal_mu,
            global_params=payload.tensors if self.proximal_mu > 0 else None,
        )
        # return only shared params (what gets aggregated)
        update = model.get_parameters(only_shared=True)
        return FitResult(
            update=Payload(kind=PayloadKind.PARAMETERS, tensors=update),
            num_examples=train_metrics.get("n", data["y"].shape[0]),
            train_metrics=train_metrics,
        )

    def aggregate(self, round_idx, results) -> None:
        params = [fr.update.tensors for _, fr in results]
        weights = [max(fr.num_examples, 1) for _, fr in results]
        new_shared = weighted_average(params, weights)
        self._global.set_parameters(new_shared, only_shared=True)


@STRATEGIES.register("fedprox")
class FedProx(FedAvg):
    name = "fedprox"

    def __init__(self, factory, dataset, mu: float = 0.1, **kwargs) -> None:
        super().__init__(factory, dataset, **kwargs)
        self.proximal_mu = float(mu)
