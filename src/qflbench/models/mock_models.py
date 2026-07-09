"""Mock multimodal model backend (numpy only) — makes the harness runnable here.

Architecture (mirrors the real multimodal design):

    for each modality m the client holds:
        h_m = relu(x_m @ W_enc[m] + b_enc[m])     # per-modality encoder -> embed_dim
    z   = mean_m h_m                               # simple fusion (mean over present mods)
    logits = z @ W_head + b_head                   # shared classifier head

Key design points that matter for the scenarios:
  - Per-modality encoder params are keyed "enc.<modality>.*" and are NOT shared.
  - Fusion is parameter-free (mean) so it trivially handles a *subset* of
    modalities -> this is the mechanism for scenario (3): a client with only
    'image' just averages over {image}. The shared head still lives in a common
    space because all encoders map to the same embed_dim.
  - "head.*" keys ARE shared -> shared_parameter_keys() returns them. This is the
    operational meaning of "same model" in scenario (3): only the head (and, when
    present, shared fusion params) is aggregated.
  - Different `hidden`/`embed_dim`/encoder widths across clients => different
    architectures (scenarios 2, 4). As long as embed_dim (hence head shape) is
    declared shared and equal, the head can still be aggregated; prototypes and
    logits work regardless of encoder shape.

Training is vanilla mini-batch SGD on softmax cross-entropy, with an optional
FedProx proximal term on the shared parameters.

This is intentionally simple. It is a BENCHMARKING SUBSTRATE to validate the
federated protocol, not a model meant to win on real data. The torch backend
swaps in real encoders behind the same interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.interfaces import FederatedDataset, ModelBackend, ModelFactory
from ..core.registry import MODEL_FACTORIES
from ..core.types import ClientContext, TensorDict


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MockMultimodalModel(ModelBackend):
    def __init__(
        self,
        ctx: ClientContext,
        dataset: FederatedDataset,
        embed_dim: int = 16,
        hidden: int = 0,           # 0 => single linear encoder layer
        share_encoders: bool = True,
        all_modalities: bool = False,
        seed: int = 0,
    ) -> None:
        self._ctx = ctx
        self._embed_dim = embed_dim
        self._num_classes = ctx.num_classes
        self._share_encoders = share_encoders
        rng = np.random.default_rng(seed + abs(ctx.client_id))

        # per-modality encoders: x_m (d_m) -> embed_dim
        # all_modalities=True (scenario 3, Saha-faithful "same model"): EVERY client
        # instantiates encoders for ALL dataset modalities, even ones absent from
        # its local data. Absent-modality encoders receive no gradient locally and
        # are returned unchanged — under full FedAvg this reproduces the dilution
        # by which unimodal clients drag the joint model (the incongruity effect).
        enc_mods = dataset.modalities if all_modalities else ctx.modalities
        self._enc: Dict[str, Dict[str, np.ndarray]] = {}
        for m in enc_mods:
            d = dataset.feature_dim(m)
            scale = 1.0 / np.sqrt(d)
            self._enc[m] = {
                "W": rng.normal(0, scale, size=(d, embed_dim)).astype(np.float32),
                "b": np.zeros(embed_dim, dtype=np.float32),
            }

        # shared classifier head: embed_dim -> num_classes
        scale = 1.0 / np.sqrt(embed_dim)
        self._head = {
            "W": rng.normal(0, scale, size=(embed_dim, self._num_classes)).astype(np.float32),
            "b": np.zeros(self._num_classes, dtype=np.float32),
        }

    # -- identification --
    @property
    def context(self) -> ClientContext:
        return self._ctx

    def embedding_dim(self) -> int:
        return self._embed_dim

    # -- parameter (de)serialization --
    def get_parameters(self, only_shared: bool = False) -> TensorDict:
        all_params: TensorDict = {
            "head.W": self._head["W"].copy(),
            "head.b": self._head["b"].copy(),
        }
        for m, p in self._enc.items():
            all_params[f"enc.{m}.W"] = p["W"].copy()
            all_params[f"enc.{m}.b"] = p["b"].copy()
        if only_shared:
            keys = set(self.shared_parameter_keys())
            return {k: v for k, v in all_params.items() if k in keys}
        return all_params

    def set_parameters(self, params: TensorDict, only_shared: bool = False) -> None:
        # load any keys that are present; the strategy controls which keys it sends.
        if "head.W" in params:
            self._head["W"] = params["head.W"].astype(np.float32).copy()
        if "head.b" in params:
            self._head["b"] = params["head.b"].astype(np.float32).copy()
        for m in self._enc:
            if f"enc.{m}.W" in params:
                self._enc[m]["W"] = params[f"enc.{m}.W"].astype(np.float32).copy()
            if f"enc.{m}.b" in params:
                self._enc[m]["b"] = params[f"enc.{m}.b"].astype(np.float32).copy()

    def shared_parameter_keys(self) -> List[str]:
        # Scenario 1 (homogeneous, full modality): share the WHOLE model so there
        # is a coherent global model (encoders + head).
        # Scenario 3 (modality subset): share only the head; per-modality encoders
        # stay local. Controlled by share_encoders.
        keys = ["head.W", "head.b"]
        if self._share_encoders:
            for m in self._enc:
                keys += [f"enc.{m}.W", f"enc.{m}.b"]
        return keys

    # -- forward --
    def _embed(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        present = [m for m in self._enc if m in x]
        if not present:
            raise ValueError("no known modality present in input")
        embs = []
        for m in present:
            h = x[m].astype(np.float32) @ self._enc[m]["W"] + self._enc[m]["b"]
            embs.append(_relu(h))
        # mean fusion over present modalities (parameter-free -> subset-friendly)
        return np.mean(np.stack(embs, axis=0), axis=0)

    def _forward(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        z = self._embed(x)
        return z @ self._head["W"] + self._head["b"]

    def predict_logits(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        return self._forward(x)

    # -- local training (SGD on cross-entropy, optional FedProx term) --
    def local_train(
        self,
        data: Dict[str, Any],
        epochs: int,
        lr: float,
        proximal_mu: float = 0.0,
        global_params: Optional[TensorDict] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        x_all = data["x"]
        y = data["y"].astype(np.int64)
        n = len(y)
        if n == 0:
            return {"loss": float("nan"), "acc": float("nan"), "n": 0}

        batch = min(64, n)
        rng = np.random.default_rng(1234 + self._ctx.client_id)
        last_loss = 0.0

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                bidx = order[start:start + batch]
                xb = {m: x_all[m][bidx] for m in x_all}
                yb = y[bidx]

                # forward
                present = [m for m in self._enc if m in xb]
                h_pre = {m: xb[m].astype(np.float32) @ self._enc[m]["W"] + self._enc[m]["b"]
                         for m in present}
                h = {m: _relu(h_pre[m]) for m in present}
                z = np.mean(np.stack([h[m] for m in present], axis=0), axis=0)
                logits = z @ self._head["W"] + self._head["b"]
                probs = _softmax(logits)

                # loss
                eps = 1e-9
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + eps))
                last_loss = float(loss)

                # backward
                dlogits = probs.copy()
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)

                gW_head = z.T @ dlogits
                gb_head = dlogits.sum(axis=0)
                dz = dlogits @ self._head["W"].T            # (b, embed)

                # FedProto regularizer: pull each embedding toward the GLOBAL
                # prototype of its class. Loss += proto_mu/2 * ||z_i - g[y_i]||^2.
                # This is the core mechanism that ALIGNS embedding spaces across
                # heterogeneous clients; without it, averaged prototypes live in
                # incompatible spaces and nearest-prototype classification fails.
                proto_mu = 0.0
                global_protos = None
                if extra is not None:
                    proto_mu = float(extra.get("proto_mu", 0.0))
                    global_protos = extra.get("global_prototypes", None)
                if proto_mu > 0 and global_protos:
                    target = np.zeros_like(z)
                    has_proto = np.zeros(len(yb), dtype=bool)
                    for i, lab in enumerate(yb):
                        gp = global_protos.get(int(lab))
                        if gp is not None:
                            target[i] = gp
                            has_proto[i] = True
                    if has_proto.any():
                        diff = (z - target) * has_proto[:, None]
                        dz = dz + proto_mu * diff / max(has_proto.sum(), 1)
                        last_loss += float(
                            0.5 * proto_mu * np.mean((diff[has_proto]) ** 2)
                        )

                dz_each = dz / len(present)                  # mean fusion grad

                # FedProx proximal term on shared params (head)
                if proximal_mu > 0 and global_params is not None:
                    if "head.W" in global_params:
                        gW_head += proximal_mu * (self._head["W"] - global_params["head.W"])
                    if "head.b" in global_params:
                        gb_head += proximal_mu * (self._head["b"] - global_params["head.b"])

                # head update
                self._head["W"] -= lr * gW_head
                self._head["b"] -= lr * gb_head

                # encoder updates (through relu)
                for m in present:
                    relu_mask = (h_pre[m] > 0).astype(np.float32)
                    dh = dz_each * relu_mask
                    gW = xb[m].astype(np.float32).T @ dh
                    gb = dh.sum(axis=0)
                    self._enc[m]["W"] -= lr * gW
                    self._enc[m]["b"] -= lr * gb

        train_acc = self.evaluate(data)["accuracy"]
        return {"loss": last_loss, "accuracy": train_acc, "n": int(n)}

    # -- knowledge distillation toward soft targets (FedMD) --
    def distill(
        self,
        x_public: Dict[str, np.ndarray],
        soft_targets: np.ndarray,
        epochs: int,
        lr: float,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Train the model so its predictions on the public set match the
        consensus soft targets. This is the FedMD alignment step. Because targets
        are class-probability vectors (a shared language), this works across
        heterogeneous architectures WITHOUT needing aligned embedding spaces —
        which is exactly why distillation sidesteps the FedProto instability.

        Implemented as cross-entropy against soft targets (KD with T=1 reduces to
        this; T>1 softens both sides). Gradient path mirrors local_train.
        """
        n = soft_targets.shape[0]
        if n == 0:
            return {"distill_loss": float("nan")}
        # normalize/soften targets
        st = np.asarray(soft_targets, dtype=np.float64)
        if temperature != 1.0:
            st = _softmax(st / temperature) if st.ndim == 2 else st
        # if targets are logits (not probs), convert
        row_sums = st.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            st = _softmax(st)

        batch = min(64, n)
        rng = np.random.default_rng(777 + self._ctx.client_id)
        last = 0.0
        present = [m for m in self._enc if m in x_public]
        if not present:
            return {"distill_loss": float("nan")}

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                bidx = order[start:start + batch]
                xb = {m: x_public[m][bidx] for m in present}
                tb = st[bidx]

                h_pre = {m: xb[m].astype(np.float32) @ self._enc[m]["W"] + self._enc[m]["b"]
                         for m in present}
                h = {m: _relu(h_pre[m]) for m in present}
                z = np.mean(np.stack([h[m] for m in present], axis=0), axis=0)
                logits = z @ self._head["W"] + self._head["b"]
                probs = _softmax(logits / temperature)

                eps = 1e-9
                last = float(-np.mean(np.sum(tb * np.log(probs + eps), axis=1)))

                # gradient of soft-target CE (with temperature scaling)
                dlogits = (probs - tb) / len(bidx) / max(temperature, 1e-6)

                gW_head = z.T @ dlogits
                gb_head = dlogits.sum(axis=0)
                dz = dlogits @ self._head["W"].T
                dz_each = dz / len(present)

                self._head["W"] -= lr * gW_head
                self._head["b"] -= lr * gb_head
                for m in present:
                    relu_mask = (h_pre[m] > 0).astype(np.float32)
                    dh = dz_each * relu_mask
                    self._enc[m]["W"] -= lr * (xb[m].astype(np.float32).T @ dh)
                    self._enc[m]["b"] -= lr * dh.sum(axis=0)
        return {"distill_loss": last}

    # -- prototypes (FedProto) --
    def class_prototypes(self, data: Dict[str, Any]) -> Dict[int, np.ndarray]:
        x_all = data["x"]
        y = data["y"].astype(np.int64)
        z = self._embed(x_all)
        protos: Dict[int, np.ndarray] = {}
        for c in np.unique(y):
            protos[int(c)] = z[y == c].mean(axis=0).astype(np.float32)
        return protos

    # -- LOOT: embedding alignment toward other clients' mean embeddings --
    def embed(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        """Public read-only access to the fused embedding (used by LOOT)."""
        return self._embed(x)

    def align_embeddings(
        self,
        x: Dict[str, np.ndarray],
        target_z: np.ndarray,
        epochs: int = 1,
        lr: float = 0.05,
    ) -> float:
        """LOOT fine-tuning step: minimize 0.5*||z(x) - target_z||^2 wrt the
        ENCODERS (head untouched), pulling this model's public-data embeddings
        toward the leave-one-out mean of the other clients' embeddings. Same math
        as the FedProto prototype pull, but applied as a standalone SGD pass on
        the (unlabeled) public set. Returns final mean alignment loss."""
        loss = 0.0
        n = max(len(target_z), 1)
        for _ in range(max(epochs, 1)):
            present = [m for m in self._enc if m in x]
            if not present:
                return 0.0
            hs = {m: x[m].astype(np.float32) @ self._enc[m]["W"] + self._enc[m]["b"]
                  for m in present}
            acts = {m: _relu(h) for m, h in hs.items()}
            z = np.mean(np.stack([acts[m] for m in present], axis=0), axis=0)
            diff = z - target_z.astype(np.float32)
            loss = float(0.5 * np.mean(diff ** 2))
            dz = diff / n
            for m in present:
                dh = (dz / len(present)) * (hs[m] > 0)   # through mean fusion + ReLU
                gW = x[m].astype(np.float32).T @ dh
                gb = dh.sum(axis=0)
                self._enc[m]["W"] -= lr * gW
                self._enc[m]["b"] -= lr * gb
        return loss

    # -- evaluation --
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        from ..metrics.classification import classification_metrics
        x_all = data["x"]
        y = data["y"].astype(np.int64)
        if len(y) == 0:
            return {"accuracy": float("nan"), "macro_f1": float("nan")}
        logits = self._forward(x_all)
        return classification_metrics(y, logits, num_classes=self._num_classes)


@MODEL_FACTORIES.register("mock")
class MockModelFactory(ModelFactory):
    """Builds mock models. Reads per-architecture sizes from ctx.extra so that
    different clients can get different widths (model heterogeneity)."""

    def __init__(self, embed_dim: int = 16, share_encoders: bool = True,
                 all_modalities: bool = False, seed: int = 0) -> None:
        self.embed_dim = embed_dim
        self.share_encoders = share_encoders
        self.all_modalities = all_modalities
        self.seed = seed

    def build(self, ctx: ClientContext, dataset: FederatedDataset) -> MockMultimodalModel:
        # embed_dim MUST match across clients for head aggregation AND prototype
        # averaging to be valid; encoder hidden width may differ freely.
        embed_dim = int(ctx.extra.get("embed_dim", self.embed_dim))
        return MockMultimodalModel(
            ctx, dataset, embed_dim=embed_dim,
            share_encoders=self.share_encoders,
            all_modalities=self.all_modalities, seed=self.seed,
        )
