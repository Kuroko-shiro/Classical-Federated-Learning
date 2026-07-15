"""Complementary two-modality synthetic dataset for harness smoke tests."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..core.interfaces import FederatedDataset
from ..core.registry import DATASETS


@DATASETS.register("synthetic")
class SyntheticMultimodalDataset(FederatedDataset):
    def __init__(
        self,
        num_classes: int = 6,
        n_per_class: int = 300,
        image_dim: int = 40,
        text_dim: int = 30,
        subclusters: int = 3,
        cluster_spread: float = 1.5,
        noise: float = 1.5,
        seed: int = 0,
    ) -> None:
        self._num_classes = int(num_classes)
        self._dims = {"image": int(image_dim), "text": int(text_dim)}
        rng = np.random.default_rng(seed)
        xs = {name: [] for name in self._dims}
        ys = []
        # Independent modality centres make both views informative while the
        # subclusters keep the task non-trivial for a linear classifier.
        centres = {
            name: rng.normal(0, cluster_spread, size=(num_classes, subclusters, dim))
            for name, dim in self._dims.items()
        }
        for label in range(num_classes):
            assignments = rng.integers(0, subclusters, size=n_per_class)
            for name, dim in self._dims.items():
                signal = centres[name][label, assignments]
                xs[name].append((signal + rng.normal(0, noise, size=(n_per_class, dim))).astype(np.float32))
            ys.append(np.full(n_per_class, label, dtype=np.int64))
        self._x = {name: np.concatenate(values, axis=0) for name, values in xs.items()}
        self._y = np.concatenate(ys, axis=0)

        train, val, test = [], [], []
        for label in range(num_classes):
            indices = np.flatnonzero(self._y == label)
            rng.shuffle(indices)
            n_train = int(round(0.70 * len(indices)))
            n_val = int(round(0.15 * len(indices)))
            train.extend(indices[:n_train])
            val.extend(indices[n_train:n_train + n_val])
            test.extend(indices[n_train + n_val:])
        self.train_pool_index = np.asarray(sorted(train), dtype=int)
        self._val_index = np.asarray(sorted(val), dtype=int)
        self._test_index = np.asarray(sorted(test), dtype=int)
        self._client_train: Dict[int, np.ndarray] = {}

    @property
    def modalities(self):
        return list(self._dims)

    @property
    def num_classes(self):
        return self._num_classes

    def feature_dim(self, modality: str) -> int:
        return self._dims[modality]

    def labels_for_partition(self):
        return self._y[self.train_pool_index]

    def attach_partition(self, client_train) -> None:
        self._client_train = {
            int(cid): np.asarray(indices, dtype=int) for cid, indices in client_train.items()
        }

    def get_split(self, client_id: int, split: str, modalities: Sequence[str]):
        if split == "train":
            if client_id not in self._client_train:
                raise KeyError(f"no train partition attached for client {client_id}")
            indices = self._client_train[client_id]
        elif split == "val":
            indices = self._val_index
        elif split == "test":
            indices = self._test_index
        else:
            raise ValueError("split must be train, val, or test")
        return {
            "x": {name: self._x[name][indices] for name in modalities},
            "y": self._y[indices],
            "index": indices.copy(),
        }

    def public_set(self, size: int, modalities: Sequence[str], rng: np.random.Generator):
        pool = np.concatenate([self._val_index, self._test_index])
        replace = int(size) > len(pool)
        indices = rng.choice(pool, size=int(size), replace=replace)
        return {
            "x": {name: self._x[name][indices] for name in modalities},
            "y": self._y[indices],
            "index": np.asarray(indices, dtype=int),
        }
