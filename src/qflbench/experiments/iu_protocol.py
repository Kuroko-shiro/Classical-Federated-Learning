"""Phase-0 protocol utilities shared by every IU X-ray runner.

This module makes the high-risk experiment choices explicit and testable:

* one frozen split file and one full 627-study test set;
* deterministic client-local validation carved from the training partitions;
* validation-only checkpoint selection (the test set is evaluated once);
* actual payload byte accounting split by direction and client;
* portable NumPy checkpoints with protocol metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np


DEFAULT_TRAIN_SUBSET = 2510
DEFAULT_TEST_SUBSET = 627
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_VAL_SEED = 0


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class IUSplit:
    public: list[int]
    train_by_client: Dict[int, list[int]]
    val_by_client: Dict[int, list[int]]
    validation: list[int]
    test: list[int]
    split_hash: str
    alpha: str
    val_fraction: float
    val_seed: int
    val_strategy: str

    @property
    def train_size(self) -> int:
        return sum(len(v) for v in self.train_by_client.values())

    @property
    def validation_size(self) -> int:
        return len(self.validation)

    def provenance(self) -> dict:
        return {
            "split_sha256": self.split_hash,
            "alpha": self.alpha,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": len(self.test),
            "public_size": len(self.public),
            "val_fraction": self.val_fraction,
            "val_seed": self.val_seed,
            "val_strategy": self.val_strategy,
            "train_client_sizes": {
                str(cid): len(values) for cid, values in self.train_by_client.items()
            },
            "val_client_sizes": {
                str(cid): len(values) for cid, values in self.val_by_client.items()
            },
        }


def _client_train_validation(
    indices: Iterable[int], val_fraction: float, seed: int,
    labels: Optional[np.ndarray] = None,
) -> tuple[list[int], list[int]]:
    values = np.asarray(sorted(int(i) for i in indices), dtype=int)
    if not len(values) or val_fraction == 0:
        return values.tolist(), []
    n_val = int(round(len(values) * val_fraction))
    if len(values) > 1:
        n_val = min(max(n_val, 1), len(values) - 1)
    else:
        n_val = 0
    if labels is not None and n_val:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=n_val, random_state=seed,
            )
            train_positions, val_positions = next(
                splitter.split(np.zeros(len(values)), labels[values])
            )
            train = sorted(values[train_positions].tolist())
            val = sorted(values[val_positions].tolist())
            return train, val
        except (ImportError, ValueError):
            pass
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    val = sorted(values[:n_val].tolist())
    train = sorted(values[n_val:].tolist())
    return train, val


def load_iu_split(
    path: str,
    *,
    manifest_size: int,
    alpha: float,
    clients: int,
    train_subset: Optional[int] = DEFAULT_TRAIN_SUBSET,
    test_subset: Optional[int] = DEFAULT_TEST_SUBSET,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    val_seed: int = DEFAULT_VAL_SEED,
    labels: Optional[np.ndarray] = None,
    all_data: bool = False,
) -> IUSplit:
    """Load, validate and derive the Phase-0 train/validation/test protocol."""

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must satisfy 0 <= value < 1")
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if int(raw["meta"]["manifest_size"]) != int(manifest_size):
        raise RuntimeError(
            f"manifest size {manifest_size} != split file "
            f"{raw['meta']['manifest_size']}; rebuild the frozen split"
        )
    alpha_key = str(float(alpha))
    if alpha_key not in raw["by_alpha"]:
        raise RuntimeError(
            f"alpha {alpha_key} not in split file (have {list(raw['by_alpha'])})"
        )

    split_clients = int(raw["meta"].get("clients", len(raw["by_alpha"][alpha_key])))
    if not all_data and clients != split_clients:
        raise ValueError(
            f"--clients={clients} does not match frozen split ({split_clients}); "
            "regenerate the split or use the Phase-0 default"
        )
    if all_data and clients != 1:
        raise ValueError("--all-data is reserved for the one-client centralized run")

    pool = [int(i) for i in raw["train_pool"]]
    if train_subset is not None:
        if train_subset <= 0:
            raise ValueError("train_subset must be positive or omitted")
        pool = pool[: min(int(train_subset), len(pool))]
    allowed = set(pool)
    source_parts = {
        int(cid): [int(i) for i in values if int(i) in allowed]
        for cid, values in raw["by_alpha"][alpha_key].items()
    }
    flattened = [index for values in source_parts.values() for index in values]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("frozen client partitions overlap")
    if set(flattened) != allowed:
        missing = len(allowed.difference(flattened))
        extra = len(set(flattened).difference(allowed))
        raise RuntimeError(
            f"frozen client partitions do not cover the selected train pool "
            f"(missing={missing}, extra={extra})"
        )
    if all_data:
        source_parts = {0: sorted({i for values in source_parts.values() for i in values})}
    else:
        source_parts = {cid: source_parts[cid] for cid in range(clients)}

    train_by_client: Dict[int, list[int]] = {}
    val_by_client: Dict[int, list[int]] = {}
    for cid, values in source_parts.items():
        train, val = _client_train_validation(
            values, val_fraction, val_seed + 1009 * cid, labels=labels,
        )
        train_by_client[cid] = train
        val_by_client[cid] = val

    validation = sorted({i for values in val_by_client.values() for i in values})
    test = [int(i) for i in raw["test"]]
    if test_subset is not None:
        if test_subset <= 0:
            raise ValueError("test_subset must be positive or omitted")
        test = test[: min(int(test_subset), len(test))]

    train_set = {i for values in train_by_client.values() for i in values}
    val_set = set(validation)
    test_set = set(test)
    public_set = {int(i) for i in raw["public"]}
    if train_set & val_set or (train_set | val_set) & test_set:
        raise RuntimeError("frozen split contains train/validation/test overlap")
    if (train_set | val_set | test_set) & public_set:
        raise RuntimeError("public distillation set overlaps a supervised split")

    return IUSplit(
        public=[int(i) for i in raw["public"]],
        train_by_client=train_by_client,
        val_by_client=val_by_client,
        validation=validation,
        test=test,
        split_hash=_sha256_json(raw),
        alpha=alpha_key,
        val_fraction=float(val_fraction),
        val_seed=int(val_seed),
        val_strategy=("multilabel_iterative_stratification" if labels is not None
                      else "deterministic_random"),
    )


def payload_nbytes(payload: object) -> int:
    """Actual numeric payload bytes, recursively, without protocol overhead."""

    if payload is None:
        return 0
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    if isinstance(payload, Mapping):
        return sum(payload_nbytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(payload_nbytes(value) for value in payload)
    if hasattr(payload, "nbytes"):
        return int(payload.nbytes)
    if hasattr(payload, "numel") and hasattr(payload, "element_size"):
        return int(payload.numel()) * int(payload.element_size())
    raise TypeError(f"cannot account payload type {type(payload).__name__}")


class CommunicationLedger:
    """Per-round, per-direction, per-client communication accounting."""

    def __init__(self) -> None:
        self._round: Optional[int] = None
        self._round_rows: MutableMapping[tuple[int, str], int] = {}
        self._serialized_round_rows: MutableMapping[tuple[int, str], int] = {}
        self._events: list[dict] = []
        self.cumulative_upload_bytes = 0
        self.cumulative_download_bytes = 0
        self.cumulative_serialized_upload_bytes = 0
        self.cumulative_serialized_download_bytes = 0

    def start_round(self, round_index: int) -> None:
        if self._round is not None:
            raise RuntimeError("finish the current communication round first")
        self._round = int(round_index)
        self._round_rows = {}
        self._serialized_round_rows = {}
        self._events = []

    def record(
        self, client_id: int, direction: str, payload: object, *,
        payload_type: str = "unspecified", metadata: Optional[Mapping[str, object]] = None,
    ) -> int:
        if self._round is None:
            raise RuntimeError("start_round must be called before record")
        if direction not in {"upload", "download"}:
            raise ValueError("direction must be 'upload' or 'download'")
        size = payload_nbytes(payload)
        from ..communication.payload import payload_metadata
        from ..communication.serializer import serialized_payload_nbytes

        serialized_size = serialized_payload_nbytes(payload)
        key = (int(client_id), direction)
        self._round_rows[key] = self._round_rows.get(key, 0) + size
        self._serialized_round_rows[key] = (
            self._serialized_round_rows.get(key, 0) + serialized_size
        )
        self._events.append({
            "client_id": int(client_id),
            "direction": direction,
            "payload_type": str(payload_type),
            "logical_tensor_bytes": size,
            "serialized_payload_bytes": serialized_size,
            **payload_metadata(payload),
            "metadata": dict(metadata or {}),
        })
        return size

    def finish_round(self) -> dict:
        if self._round is None:
            raise RuntimeError("no active communication round")
        upload = sum(v for (cid, direction), v in self._round_rows.items() if direction == "upload")
        download = sum(v for (cid, direction), v in self._round_rows.items() if direction == "download")
        serialized_upload = sum(
            v for (cid, direction), v in self._serialized_round_rows.items()
            if direction == "upload"
        )
        serialized_download = sum(
            v for (cid, direction), v in self._serialized_round_rows.items()
            if direction == "download"
        )
        self.cumulative_upload_bytes += upload
        self.cumulative_download_bytes += download
        self.cumulative_serialized_upload_bytes += serialized_upload
        self.cumulative_serialized_download_bytes += serialized_download
        clients = sorted({cid for cid, _ in self._round_rows})
        from ..communication.qkd_accounting import qkd_key_budget

        result = {
            "round": self._round,
            "upload_bytes": upload,
            "download_bytes": download,
            "total_bytes": upload + download,
            "cumulative_upload_bytes": self.cumulative_upload_bytes,
            "cumulative_download_bytes": self.cumulative_download_bytes,
            "cumulative_total_bytes": self.cumulative_upload_bytes + self.cumulative_download_bytes,
            "serialized_upload_bytes": serialized_upload,
            "serialized_download_bytes": serialized_download,
            "serialized_total_bytes": serialized_upload + serialized_download,
            "cumulative_serialized_upload_bytes": self.cumulative_serialized_upload_bytes,
            "cumulative_serialized_download_bytes": self.cumulative_serialized_download_bytes,
            "cumulative_serialized_total_bytes": (
                self.cumulative_serialized_upload_bytes
                + self.cumulative_serialized_download_bytes
            ),
            "clients": {
                str(cid): {
                    "upload_bytes": self._round_rows.get((cid, "upload"), 0),
                    "download_bytes": self._round_rows.get((cid, "download"), 0),
                    "serialized_upload_bytes": self._serialized_round_rows.get((cid, "upload"), 0),
                    "serialized_download_bytes": self._serialized_round_rows.get((cid, "download"), 0),
                }
                for cid in clients
            },
            "payloads": list(self._events),
            "qkd_otp": qkd_key_budget(serialized_upload + serialized_download),
        }
        self._round = None
        self._round_rows = {}
        self._serialized_round_rows = {}
        self._events = []
        return result


def _normalise_models(models: Mapping[str, Mapping[str, object]]) -> Dict[str, np.ndarray]:
    flat: Dict[str, np.ndarray] = {}
    for model_name, params in models.items():
        for param_name, value in params.items():
            flat[f"model::{model_name}::{param_name}"] = np.asarray(value)
    return flat


class BestCheckpoint:
    """Persist only the best validation checkpoint, atomically."""

    def __init__(self, path: str, *, metric: str = "auroc", mode: str = "max"):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be max or min")
        self.path = Path(path)
        self.metric = metric
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_round: Optional[int] = None

    def update(
        self,
        round_index: int,
        metrics: Mapping[str, float],
        models: Mapping[str, Mapping[str, object]],
        *,
        metadata: Optional[Mapping[str, object]] = None,
        arrays: Optional[Mapping[str, object]] = None,
    ) -> bool:
        score = float(metrics[self.metric])
        improved = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if not improved:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_meta = dict(metadata or {})
        checkpoint_meta.update({
            "selection_metric": self.metric,
            "selection_mode": self.mode,
            "best_round": int(round_index),
            "best_validation_metrics": {k: float(v) for k, v in metrics.items()},
        })
        payload = _normalise_models(models)
        for key, value in (arrays or {}).items():
            payload[f"array::{key}"] = np.asarray(value)
        payload["__metadata_json__"] = np.frombuffer(
            json.dumps(checkpoint_meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **payload)
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        self.best_score = score
        self.best_round = int(round_index)
        return True


def load_checkpoint(path: str) -> tuple[dict, Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Load a :class:`BestCheckpoint` artifact."""

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["__metadata_json__"].tolist()).decode("utf-8"))
        models: Dict[str, Dict[str, np.ndarray]] = {}
        arrays: Dict[str, np.ndarray] = {}
        for key in archive.files:
            if key.startswith("model::"):
                _, model_name, param_name = key.split("::", 2)
                models.setdefault(model_name, {})[param_name] = archive[key].copy()
            elif key.startswith("array::"):
                arrays[key.split("::", 1)[1]] = archive[key].copy()
    return metadata, models, arrays
