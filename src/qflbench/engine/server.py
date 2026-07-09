"""The federated engine: orchestrates rounds over strategy + channel + clients.

One round:
    for each selected client:
        payload   = strategy.broadcast(round, client.ctx)
        payload'  = channel.transmit(payload, DOWNLINK)     # cost accounting
        fitres    = strategy.client_fit(client.model, payload', train_data, cfg)
        _         = channel.transmit(fitres.update, UPLINK) # cost accounting
    strategy.aggregate(round, results)
    metrics   = evaluate(round)

Evaluation:
  - if strategy.global_model() is not None -> evaluate that single model on the
    shared test set (global view), plus per-client test.
  - else (e.g. FedProto) -> per-client evaluation using each client's encoder with
    the global prototypes (nearest-prototype), since there is no single global net.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.interfaces import CommunicationChannel, FederatedStrategy
from ..core.types import (
    Direction,
    FitResult,
    RoundMetrics,
)
from ..metrics.classification import aggregate_client_metrics, classification_metrics
from .client import Client


class FederatedEngine:
    def __init__(
        self,
        strategy: FederatedStrategy,
        channel: CommunicationChannel,
        clients: List[Client],
        rng: np.random.Generator,
        client_fraction: float = 1.0,
        local_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy = strategy
        self.channel = channel
        self.clients = clients
        self.rng = rng
        self.client_fraction = client_fraction
        self.local_cfg = local_cfg or {}

    def _select(self) -> List[Client]:
        k = max(1, int(round(self.client_fraction * len(self.clients))))
        idx = self.rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in idx]

    def run_round(self, round_idx: int) -> RoundMetrics:
        selected = self._select()
        results: List[Tuple[Any, FitResult]] = []
        up_bytes = down_bytes = up_scalars = down_scalars = 0

        for client in selected:
            payload = self.strategy.broadcast(round_idx, client.ctx)
            payload = self.channel.transmit(
                payload, Direction.DOWNLINK, round_idx, client.ctx.client_id
            )
            down_bytes += payload.nbytes()
            down_scalars += payload.num_scalars()

            fitres = self.strategy.client_fit(
                client.model, payload, client.train_data(), self.local_cfg
            )

            self.channel.transmit(
                fitres.update, Direction.UPLINK, round_idx, client.ctx.client_id
            )
            up_bytes += fitres.update.nbytes()
            up_scalars += fitres.update.num_scalars()

            results.append((client.ctx, fitres))

        self.strategy.aggregate(round_idx, results)

        rm = self._evaluate(round_idx)
        rm.uplink_bytes = up_bytes
        rm.downlink_bytes = down_bytes
        rm.uplink_scalars = up_scalars
        rm.downlink_scalars = down_scalars
        return rm

    # ---- evaluation ----
    def _evaluate(self, round_idx: int) -> RoundMetrics:
        rm = RoundMetrics(round_idx=round_idx)
        gmodel = self.strategy.global_model()

        if gmodel is not None:
            # global view: shared params loaded into a full-modality model; eval on
            # the shared test split over all modalities.
            test = self.clients[0].dataset.get_split(
                self.clients[0].ctx.client_id, "test", gmodel.context.modalities
            )
            # use the shared test set (same indices for everyone); modalities here
            # are the global model's full set.
            full_mods = gmodel.context.modalities
            # rebuild a test set over full modalities from the dataset directly:
            ds = self.clients[0].dataset
            # any client's "test" split returns the shared test indices; we just
            # need all modalities present -> ask for full_mods.
            test_full = ds.get_split(self.clients[0].ctx.client_id, "test", full_mods)
            rm.global_metrics = gmodel.evaluate(test_full)

        # per-client evaluation (always useful: fairness / worst-client)
        per_client: Dict[int, Dict[str, float]] = {}
        protos = getattr(self.strategy, "global_prototypes", None)
        global_protos = protos() if callable(protos) else None

        for client in self.clients:
            test = client.test_data()
            if global_protos:
                m = _nearest_prototype_metrics(client.model, test, global_protos)
            else:
                m = client.model.evaluate(test)
            per_client[client.ctx.client_id] = m

        rm.per_client_metrics = per_client
        rm.extra = aggregate_client_metrics(per_client)
        return rm


def _nearest_prototype_metrics(model, data, global_protos) -> Dict[str, float]:
    """Classify test points by nearest GLOBAL prototype in the client's embedding
    space (FedProto-style eval)."""
    x = data["x"]
    y = data["y"].astype(int)
    if len(y) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    z = model._embed(x)  # client encoder
    classes = sorted(global_protos.keys())
    P = np.stack([global_protos[c] for c in classes], axis=0)  # (C, d)
    # squared euclidean distances -> nearest prototype
    d2 = ((z[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    pred_idx = d2.argmin(axis=1)
    y_pred = np.array([classes[i] for i in pred_idx])
    # build pseudo-logits (negative distances) for metric fn
    logits = -d2
    num_classes = max(classes) + 1
    # remap logits columns to class indices
    full = np.full((len(y), num_classes), -1e9)
    for j, c in enumerate(classes):
        full[:, c] = logits[:, j]
    return classification_metrics(y, full, num_classes=num_classes)
