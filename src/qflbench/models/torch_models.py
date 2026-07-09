"""Torch multimodal model backend — STUB (runs in the lab env, needs torch).

Implements the SAME ModelBackend interface as MockMultimodalModel, but with real
encoders. Swapping mock -> torch is then a config change (model factory name).

Suggested concrete encoders for IU X-ray:
  - image: DenseNet121 / a biomedical CNN -> global-pool -> Linear(embed_dim)
  - text:  ClinicalBERT / CXR-BERT CLS    -> Linear(embed_dim)
Fusion: mean (subset-friendly) or a small attention block (shared, aggregatable).
Head:   Linear(embed_dim -> num_classes), shared.

Implementation checklist:
  [ ] get/set_parameters via state_dict <-> numpy (detach().cpu().numpy())
  [ ] shared_parameter_keys: head (+ shared fusion) parameter names
  [ ] local_train: standard torch training loop; FedProx term = mu/2 * ||w - w_g||^2
      on shared params; SCAFFOLD/MOON hooks via `extra`
  [ ] class_prototypes: forward encoders+fusion, average embeddings per class
  [ ] predict_logits: forward on a public batch (no grad)
  [ ] seeding: torch.manual_seed in factory
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.interfaces import FederatedDataset, ModelBackend, ModelFactory
from ..core.registry import MODEL_FACTORIES
from ..core.types import ClientContext, TensorDict


class TorchMultimodalModel(ModelBackend):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TorchMultimodalModel is a stub. Implement in the lab env with torch. "
            "Mirror MockMultimodalModel's interface exactly so the harness is reused."
        )


@MODEL_FACTORIES.register("torch")
class TorchModelFactory(ModelFactory):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def build(self, ctx: ClientContext, dataset: FederatedDataset) -> ModelBackend:
        raise NotImplementedError("Provide torch in the lab environment.")
