"""Core data types shared across the federated-learning protocol.

Design rationale
----------------
The single most important architectural decision in this benchmark is that the
federated protocol is expressed in terms of *abstract payloads*, NOT in terms of
"model parameters". This is what makes scenarios (2)/(3)/(4) expressible at all:

    - FedAvg / FedProx broadcast & aggregate **parameters**.
    - FedProto broadcast & aggregate **class prototypes** (mean embeddings).
    - FedMD / FedDF broadcast & aggregate **logits on a public/proxy set**.

All three fit the same `BroadcastPayload -> (local work) -> ClientUpdate -> aggregate`
shape. If we had hard-coded "aggregate parameters" we could only ever implement
scenario (1). Everything here is deliberately framework-agnostic (no torch import)
so the harness can be validated with the numpy mock backend and later swapped to
a real torch backend without touching protocol code.

A `Payload` is just a dict of named numeric tensors plus metadata. We represent
tensors as numpy arrays at the protocol boundary. Real models may hold torch
tensors internally, but they expose/consume numpy at the protocol edge via the
ModelBackend interface (see models/base.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

# A named bag of numeric tensors. Keys are semantic (e.g. parameter names,
# class indices for prototypes, sample ids for logits). Values are numpy arrays.
TensorDict = Dict[str, np.ndarray]


class Direction(str, Enum):
    """Direction of a transmission across the communication channel."""

    DOWNLINK = "downlink"  # server -> client (broadcast)
    UPLINK = "uplink"      # client -> server (update)


class PayloadKind(str, Enum):
    """What kind of information a payload carries.

    This lets the communication channel and the metrics layer reason about the
    payload without knowing the concrete strategy. It is also the hook the future
    quantum channel will use to decide how something is encoded/transmitted
    (e.g. parameters via amplitude encoding vs. prototypes via a different scheme).
    """

    PARAMETERS = "parameters"      # model weights / deltas (FedAvg, FedProx, ...)
    PROTOTYPES = "prototypes"      # per-class mean embeddings (FedProto)
    LOGITS = "logits"              # predictions on a shared/public set (FedMD, FedDF)
    CONTROL = "control"            # control variates (SCAFFOLD), aux state
    EMPTY = "empty"                # no-op / placeholder


@dataclass
class Payload:
    """A unit of information exchanged between server and a client.

    `tensors` is the numeric content whose size determines communication cost.
    `meta` carries non-numeric scalars (counts, flags) that are not "charged" the
    same way (or charged trivially) by the communication channel.
    """

    kind: PayloadKind
    tensors: TensorDict = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- size accounting (used by the communication channel) ----
    def num_scalars(self) -> int:
        """Total number of scalar values in the payload's tensors."""
        return int(sum(int(np.asarray(v).size) for v in self.tensors.values()))

    def nbytes(self, dtype_bytes: int = 4) -> int:
        """Approximate transmitted size in bytes.

        Default assumes fp32 (4 bytes). The classical channel uses this to report
        a comparable "bytes on the wire" number. The quantum channel will instead
        report qubits/ebits, but will read the same `tensors` to size its encoding,
        which is exactly why payloads are kind-tagged and numeric.
        """
        return self.num_scalars() * dtype_bytes

    def copy(self) -> "Payload":
        return Payload(
            kind=self.kind,
            tensors={k: np.array(v, copy=True) for k, v in self.tensors.items()},
            meta=dict(self.meta),
        )

    @staticmethod
    def empty() -> "Payload":
        return Payload(kind=PayloadKind.EMPTY, tensors={}, meta={})


# Semantic aliases. Same structure, but naming the role at call sites improves
# readability of the Strategy protocol (server broadcasts a BroadcastPayload,
# clients return a ClientUpdate).
BroadcastPayload = Payload
ClientUpdate = Payload


@dataclass
class ClientContext:
    """Static description of a client, fixed for an experiment run.

    This is the object that encodes *heterogeneity*:
      - `model_name` differing across clients  => model heterogeneity (scenarios 2, 4)
      - `modalities` differing across clients   => modality heterogeneity (scenarios 3, 4)
      - data partition (held elsewhere) differing => statistical heterogeneity (all)
    """

    client_id: int
    model_name: str                      # which architecture this client builds
    modalities: List[str]                # which modalities this client actually holds
    num_train: int                       # local training set size (for weighting)
    num_classes: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransmissionRecord:
    """One logged transmission event (for communication accounting)."""

    round_idx: int
    client_id: Optional[int]
    direction: Direction
    kind: PayloadKind
    num_scalars: int
    nbytes: int
    # Reserved for the quantum channel; left None/0 in the classical setting so
    # that the same record schema is reused end-to-end.
    qubits: int = 0
    ebits: int = 0


@dataclass
class RoundMetrics:
    """Metrics captured at the end of a federated round."""

    round_idx: int
    # global model evaluation (single shared model) — may be None for purely
    # personalized strategies that have no single global model to evaluate.
    global_metrics: Optional[Dict[str, float]] = None
    # per-client evaluation on each client's local test split.
    per_client_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # communication this round.
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    uplink_scalars: int = 0
    downlink_scalars: int = 0
    # free-form extras (e.g. proxy-set accuracy for FedMD)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.uplink_bytes + self.downlink_bytes


@dataclass
class FitResult:
    """Result of a client's local training in a round."""

    update: ClientUpdate
    num_examples: int
    train_metrics: Dict[str, float] = field(default_factory=dict)
