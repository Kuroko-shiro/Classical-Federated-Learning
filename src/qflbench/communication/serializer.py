"""Reference serialization used for reproducible on-wire byte estimates."""

from __future__ import annotations

import json
from typing import Mapping


def _schema(value: object):
    if isinstance(value, Mapping):
        return {str(key): _schema(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return {"container": type(value).__name__, "items": [_schema(item) for item in value]}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "length": len(value)}
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if dtype is not None and shape is not None:
        return {
            "type": "tensor", "dtype": str(dtype),
            "shape": [int(size) for size in shape],
        }
    return {"type": type(value).__name__, "value": repr(value)}


def _logical_nbytes(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(_logical_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_logical_nbytes(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel()) * int(value.element_size())
    return 0


def serialized_payload_nbytes(payload: object) -> int:
    """Return bytes emitted by the canonical tensor serializer.

    The wire format is an unsigned 64-bit header length, canonical compact JSON
    schema, then contiguous raw tensor buffers. This makes the count exact while
    avoiding allocation/copy of a 500 MB model merely to measure it. Transport,
    TLS and packet framing are deliberately excluded.
    """

    header = json.dumps(
        _schema(payload), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return 8 + len(header) + _logical_nbytes(payload)
