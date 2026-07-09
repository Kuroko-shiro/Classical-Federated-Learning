"""Abstract interfaces (the pluggable seams of the benchmark).

Each ABC here is a place where a concrete implementation is swapped in via config.
The four seams that matter for the research plan:

  1. FederatedDataset / Partitioner  -> data-agnostic, modality-agnostic (decision D2)
  2. ModelBackend                     -> per-client architectures (scenarios 2, 4)
  3. FederatedStrategy                -> FedAvg / FedProto / FedMD / ... (the protocol)
  4. CommunicationChannel             -> classical now, QUANTUM later (the QFL swap point)

Keeping these abstract + numpy-at-the-edge is what lets us (a) run the whole thing
here with a numpy mock, and (b) later replace the model backend with torch and the
channel with a quantum-internet model, without rewriting protocol logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import (
    BroadcastPayload,
    ClientContext,
    ClientUpdate,
    Direction,
    FitResult,
    Payload,
    TensorDict,
    TransmissionRecord,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class FederatedDataset(ABC):
    """A multimodal dataset that can be queried per client and per modality.

    Implementations must be able to return, for a given client, the data
    restricted to the modalities that client holds. This is the mechanism behind
    the modality-subset interpretation of scenarios (3) and (4): the *same*
    paired multimodal dataset is used, but each client only sees some modalities.
    """

    @property
    @abstractmethod
    def modalities(self) -> List[str]:
        """All modalities available in this dataset, e.g. ['image', 'text']."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @abstractmethod
    def get_split(
        self, client_id: int, split: str, modalities: Sequence[str]
    ) -> Dict[str, Any]:
        """Return a client's data for `split` ('train'/'val'/'test'), restricted
        to `modalities`. Returns a dict like
            {"x": {modality: array, ...}, "y": labels}
        Concrete shape of each modality array is backend-specific (features for
        the mock backend; raw/encoded tensors for torch backends).
        """

    @abstractmethod
    def feature_dim(self, modality: str) -> int:
        """Dimensionality the model should expect for a given modality."""


class Partitioner(ABC):
    """Splits the global dataset index space across clients.

    Encodes statistical heterogeneity: IID, Dirichlet label skew, quantity skew.
    Modality assignment (which client holds which modalities) is handled by the
    scenario builder, not here, because it is structural rather than statistical.
    """

    @abstractmethod
    def partition(
        self, labels: np.ndarray, num_clients: int, rng: np.random.Generator
    ) -> List[np.ndarray]:
        """Return a list (length num_clients) of index arrays into the dataset."""


# --------------------------------------------------------------------------- #
# Model backend
# --------------------------------------------------------------------------- #
class ModelBackend(ABC):
    """A client-side model, abstracted so the protocol never imports torch.

    The protocol only ever needs to:
      - push/pull parameters as a TensorDict  (parameter-based strategies)
      - train locally for E epochs            (all strategies)
      - produce embeddings / prototypes        (FedProto)
      - produce logits on a given input batch  (FedMD / FedDF)
      - evaluate

    A model is built from a ClientContext, so different clients can build
    different architectures (model heterogeneity) and different input modalities
    (modality heterogeneity). The mock backend (numpy) implements all of this for
    harness validation; the torch backend (TODO, runs in the user's env)
    implements the same interface.

    `shared_parameter_keys()` is the crucial hook for scenario (3): a model can
    declare that only its fusion+head parameters are "shared" (aggregatable),
    while per-modality encoders stay local. Parameter-based strategies aggregate
    only the shared keys.
    """

    # ---- identification ----
    @property
    @abstractmethod
    def context(self) -> ClientContext:
        ...

    # ---- parameters (for parameter-based strategies) ----
    @abstractmethod
    def get_parameters(self, only_shared: bool = False) -> TensorDict:
        ...

    @abstractmethod
    def set_parameters(self, params: TensorDict, only_shared: bool = False) -> None:
        ...

    @abstractmethod
    def shared_parameter_keys(self) -> List[str]:
        """Subset of parameter keys that are shareable across clients.

        For a fully-homogeneous model this is all keys. For the modality-subset
        setting it is typically the fusion + classifier head keys only.
        """

    # ---- local training ----
    @abstractmethod
    def local_train(
        self,
        data: Dict[str, Any],
        epochs: int,
        lr: float,
        proximal_mu: float = 0.0,
        global_params: Optional[TensorDict] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Train in-place on local `data`. Returns train metrics.

        `proximal_mu` + `global_params` support FedProx's proximal term without a
        dedicated subclass. `extra` carries strategy-specific knobs (e.g. SCAFFOLD
        control variates, MOON's previous/global representations).
        """

    # ---- representations / outputs (for distillation & prototype strategies) ----
    @abstractmethod
    def class_prototypes(self, data: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """Return {class_index: mean_embedding} over local data (FedProto)."""

    @abstractmethod
    def predict_logits(self, x: Dict[str, Any]) -> np.ndarray:
        """Return logits for a batch of inputs (FedMD/FedDF on a public set)."""

    def distill(
        self,
        x_public: Dict[str, Any],
        soft_targets: np.ndarray,
        epochs: int,
        lr: float,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Align predictions on a public set to consensus soft targets (FedMD).

        Default raises; concrete backends implement it. Not abstract so that
        prototype-only backends need not provide it.
        """
        raise NotImplementedError

    # ---- evaluation ----
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        ...

    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the shared embedding space (for prototypes)."""


class ModelFactory(ABC):
    """Builds a ModelBackend for a given client context.

    The factory is what a config selects. The same factory can produce different
    architectures depending on ctx.model_name, enabling model heterogeneity.
    """

    @abstractmethod
    def build(self, ctx: ClientContext, dataset: FederatedDataset) -> ModelBackend:
        ...


# --------------------------------------------------------------------------- #
# Communication channel  (the QUANTUM swap point)
# --------------------------------------------------------------------------- #
class CommunicationChannel(ABC):
    """Transports payloads between server and clients and accounts for cost.

    THIS IS THE QFL INTEGRATION POINT. The classical channel passes tensors
    through unchanged and records bytes/scalars. A future quantum channel will
    implement the same `transmit` method but:
      - size its encoding in qubits/ebits,
      - optionally perturb payloads to model entanglement-generation failure,
        memory decoherence, and re-transmission,
      - fill in TransmissionRecord.qubits / .ebits.

    Because the protocol only talks to this interface, none of that requires
    touching Strategy or Engine code.
    """

    @abstractmethod
    def transmit(
        self, payload: Payload, direction: Direction, round_idx: int,
        client_id: Optional[int] = None,
    ) -> Payload:
        """Return the (possibly perturbed) payload as received on the other end,
        while recording a TransmissionRecord internally."""

    @abstractmethod
    def records(self) -> List[TransmissionRecord]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# Strategy  (the federated protocol)
# --------------------------------------------------------------------------- #
class FederatedStrategy(ABC):
    """Defines one federated algorithm as a broadcast/aggregate protocol.

    The lifecycle per round (orchestrated by the Engine):

        payloads   = strategy.broadcast(round_idx, selected_clients)   # downlink
        # engine sends each payload through the channel to its client,
        # asks the client model to do local work, gets a ClientUpdate back,
        # sends it through the channel (uplink)
        new_state  = strategy.aggregate(round_idx, fit_results)

    Different strategies implement these differently:
      - FedAvg/FedProx: broadcast global params; aggregate = weighted mean of params.
      - FedProto:       broadcast global prototypes; aggregate = mean of prototypes.
      - FedMD/FedDF:    broadcast consensus logits on a public set; aggregate logits.

    The strategy owns the global state. The engine owns orchestration and the
    channel. Clients own their models and data.
    """

    name: str = "abstract"

    @abstractmethod
    def initialize(self, clients: List[ClientContext]) -> None:
        """Set up global state given the participating clients."""

    @abstractmethod
    def broadcast(
        self, round_idx: int, client: ClientContext
    ) -> BroadcastPayload:
        """Produce the payload to send down to a specific client this round."""

    @abstractmethod
    def client_fit(
        self,
        model: ModelBackend,
        payload: BroadcastPayload,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> FitResult:
        """Run the client-side computation for this strategy and return an update.

        Lives on the strategy (not the client) because *what* the client computes
        and returns is part of the algorithm definition (params vs prototypes vs
        logits). The model is the substrate; the strategy decides how to use it.
        """

    @abstractmethod
    def aggregate(
        self, round_idx: int, results: List[Tuple[ClientContext, FitResult]]
    ) -> None:
        """Update global state from client updates."""

    # Optional: a single global model to evaluate. Personalized/prototype-only
    # strategies may return None (the engine then relies on per-client eval).
    def global_model(self) -> Optional[ModelBackend]:
        return None

    def needs_proxy_data(self) -> bool:
        """True for distillation strategies that require a public/proxy set."""
        return False
