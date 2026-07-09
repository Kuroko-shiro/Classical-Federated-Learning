"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python/numpy RNGs and return a numpy Generator for explicit use.

    Prefer threading the returned Generator through the code (partitioner, model
    init) rather than relying on global state, so that runs are reproducible even
    when components are reordered.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch seeding will be added in the torch backend:
    #   torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
