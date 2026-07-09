"""A tiny registry so configs can select components by name.

Each pluggable family (strategy, model factory, channel, partitioner, dataset)
registers concrete classes under string keys. Hydra/config passes a string; the
registry resolves it to a class. This keeps the swap points declarative.

Usage:
    from qflbench.core.registry import STRATEGIES

    @STRATEGIES.register("fedavg")
    class FedAvg(FederatedStrategy): ...

    strat_cls = STRATEGIES.get("fedavg")
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._table: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._table:
                raise KeyError(f"{self._kind} '{name}' already registered")
            self._table[name] = cls
            return cls
        return deco

    def get(self, name: str) -> Type[T]:
        if name not in self._table:
            raise KeyError(
                f"unknown {self._kind} '{name}'. "
                f"available: {sorted(self._table)}"
            )
        return self._table[name]

    def available(self):
        return sorted(self._table)


# One registry per pluggable family.
STRATEGIES: Registry = Registry("strategy")
MODEL_FACTORIES: Registry = Registry("model_factory")
CHANNELS: Registry = Registry("channel")
PARTITIONERS: Registry = Registry("partitioner")
DATASETS: Registry = Registry("dataset")
