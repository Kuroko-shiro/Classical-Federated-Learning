"""Dependency-light IID and non-IID client partitioners."""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.interfaces import Partitioner
from ..core.registry import PARTITIONERS


def _repair_empty(parts: List[list[int]]) -> None:
    for empty in [index for index, values in enumerate(parts) if not values]:
        donor = max(range(len(parts)), key=lambda index: len(parts[index]))
        if len(parts[donor]) <= 1:
            break
        parts[empty].append(parts[donor].pop())


@PARTITIONERS.register("iid")
class IIDPartitioner(Partitioner):
    def partition(self, labels, num_clients, rng):
        indices = rng.permutation(len(labels))
        return [np.sort(chunk.astype(int)) for chunk in np.array_split(indices, num_clients)]


@PARTITIONERS.register("dirichlet")
class DirichletPartitioner(Partitioner):
    def __init__(self, alpha: float = 0.5) -> None:
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        self.alpha = float(alpha)

    def partition(self, labels, num_clients, rng):
        y = np.asarray(labels)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        parts: List[list[int]] = [[] for _ in range(num_clients)]
        for label in sorted(np.unique(y).tolist()):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, self.alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for cid, chunk in enumerate(np.split(indices, cuts)):
                parts[cid].extend(int(index) for index in chunk)
        _repair_empty(parts)
        return [np.asarray(sorted(values), dtype=int) for values in parts]


@PARTITIONERS.register("quantity_skew")
class QuantitySkewPartitioner(Partitioner):
    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(sigma)

    def partition(self, labels, num_clients, rng):
        indices = rng.permutation(len(labels))
        proportions = rng.lognormal(0.0, self.sigma, size=num_clients)
        proportions /= proportions.sum()
        cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        parts = [list(map(int, chunk)) for chunk in np.split(indices, cuts)]
        _repair_empty(parts)
        return [np.asarray(sorted(values), dtype=int) for values in parts]
