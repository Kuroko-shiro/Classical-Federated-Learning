"""FedMD — distillation-based strategy for MODEL heterogeneity (scenario 2, alt).

Protocol (Li & Wang, 2019):
  1. server holds a PUBLIC/proxy set (unlabeled by convention);
  2. each client predicts logits on the public set and uploads them;
  3. server averages -> consensus logits, broadcasts them down;
  4. each client distills toward the consensus on the public set, then trains on
     its private data.

Communication = (public_size x num_classes) scalars per direction. Unlike FedProto
it requires a shared public set (needs_proxy_data() -> True), which is the main
practical cost. We include it as the distillation baseline to contrast with
FedProto's proxy-free, lower-communication approach.

This skeleton implements the consensus-logit exchange and the bookkeeping. The
actual distillation step in `client_fit` is marked TODO in the mock backend
(needs a soft-target training path); FedProto is the runnable scenario-2 method in
the smoke test, while FedMD's wiring/accounting is in place for the torch backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.interfaces import FederatedStrategy, ModelFactory
from ..core.registry import STRATEGIES
from ..core.types import (
    BroadcastPayload,
    ClientContext,
    FitResult,
    Payload,
    PayloadKind,
)


@STRATEGIES.register("fedmd")
class FedMD(FederatedStrategy):
    name = "fedmd"

    def __init__(self, factory: ModelFactory, dataset, public_size: int = 200,
                 distill_epochs: int = 1, temperature: float = 1.0, **kwargs) -> None:
        self.factory = factory
        self.dataset = dataset
        self.public_size = int(public_size)
        self.distill_epochs = int(distill_epochs)
        self.temperature = float(temperature)
        self._public: Optional[Dict[str, Any]] = None
        self._consensus: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(0)

    def needs_proxy_data(self) -> bool:
        return True

    def initialize(self, clients: List[ClientContext]) -> None:
        # build a public set over ALL modalities; per-client we will restrict to
        # the modalities each client holds when it predicts.
        if hasattr(self.dataset, "public_set"):
            self._public = self.dataset.public_set(
                self.public_size, self.dataset.modalities, self._rng
            )
        else:
            raise RuntimeError("dataset does not expose a public_set() for FedMD")
        self._consensus = None

    def broadcast(self, round_idx, client) -> BroadcastPayload:
        tensors = {}
        if self._consensus is not None:
            tensors["consensus_logits"] = self._consensus
        return Payload(kind=PayloadKind.LOGITS, tensors=tensors,
                       meta={"round": round_idx, "public_index": self._public["index"]})

    def client_fit(self, model, payload, data, config) -> FitResult:
        client_mods = model.context.modalities
        pub_x = {m: self._public["x"][m] for m in client_mods if m in self._public["x"]}

        # 1) distill toward the consensus on the public set (the alignment step).
        #    Targets are class-probability vectors, so this works across different
        #    architectures without aligned embeddings (unlike FedProto).
        if "consensus_logits" in payload.tensors:
            model.distill(
                pub_x,
                payload.tensors["consensus_logits"],
                epochs=self.distill_epochs,
                lr=config.get("distill_lr", config.get("lr", 0.05)),
                temperature=self.temperature,
            )

        # 2) train on private (labeled) data
        train_metrics = model.local_train(
            data,
            epochs=config.get("local_epochs", 1),
            lr=config.get("lr", 0.05),
        )

        # 3) predict logits on the public set to contribute to next consensus
        logits = model.predict_logits(pub_x)
        return FitResult(
            update=Payload(kind=PayloadKind.LOGITS, tensors={"logits": logits}),
            num_examples=train_metrics.get("n", data["y"].shape[0]),
            train_metrics=train_metrics,
        )

    def aggregate(self, round_idx, results) -> None:
        stacks = [fr.update.tensors["logits"] for _, fr in results
                  if "logits" in fr.update.tensors]
        if stacks:
            self._consensus = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)

    def global_model(self):
        return None
