"""Describe numeric payloads without retaining their contents."""

from __future__ import annotations

from collections import Counter
from typing import Mapping


def _visit(value: object, dtypes: Counter, tensors: list[dict]) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _visit(item, dtypes, tensors)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _visit(item, dtypes, tensors)
        return
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if dtype is not None and shape is not None:
        name = str(dtype)
        dtypes[name] += 1
        tensors.append({"dtype": name, "shape": [int(size) for size in shape]})


def payload_metadata(payload: object) -> dict:
    dtypes: Counter = Counter()
    tensors: list[dict] = []
    _visit(payload, dtypes, tensors)
    return {
        "tensor_count": len(tensors),
        "dtypes": dict(sorted(dtypes.items())),
        "shapes": tensors,
    }
