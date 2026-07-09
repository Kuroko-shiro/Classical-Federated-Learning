"""Strategy base helpers.

Concrete strategies live in fedavg.py, fedprox.py, fedproto.py, fedmd.py, ...
This module provides a base for the common "single global model" pattern used by
parameter-aggregation strategies (FedAvg, FedProx, and—via shared head only—the
scenario-3 variants).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.interfaces import FederatedStrategy, ModelBackend, ModelFactory
from ..core.types import ClientContext


class SingleGlobalModelStrategy(FederatedStrategy):
    """Base for strategies that maintain one global ModelBackend.

    Subclasses set `self._global` in `initialize` and implement broadcast/fit/
    aggregate. We hold a model factory + a reference dataset so the global model
    can be (re)built and evaluated centrally.
    """

    def __init__(self, factory: ModelFactory, dataset, **kwargs) -> None:
        self.factory = factory
        self.dataset = dataset
        self._global: Optional[ModelBackend] = None

    def global_model(self) -> Optional[ModelBackend]:
        return self._global
