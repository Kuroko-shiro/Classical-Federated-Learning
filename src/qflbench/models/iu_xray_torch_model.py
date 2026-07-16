"""Torch multimodal model for IU X-ray (method B), ModelBackend-compatible.

Architecture (Saha-style):
  image encoder : ResNet-50 (ImageNet-pretrained) -> Linear(embed_dim)
  text encoder  : BERT-base  (pretrained)         -> Linear(embed_dim)  [CLS]
  fusion        : mean over present modalities (parameter-free; subset-friendly,
                  so the SAME head works whether a client has image, text, or both)
  head          : Linear(embed_dim -> 14)  shared, multi-label (BCEWithLogits)

Implements the ModelBackend contract used by the strategies:
  get/set_parameters(only_shared) | shared_parameter_keys | local_train |
  class_prototypes | predict_logits | distill | align_embeddings | embed | evaluate

Flags:
  share_encoders : ① True (encoders aggregated) / ③④ False (encoders local, only
                   fusion+head shared) — same semantics as the numpy backend.
  all_modalities : ③ Saha-faithful "same model" — every client instantiates BOTH
                   encoders even if its data lacks a modality.

Heavy on the M5; tips: keep batch size modest (e.g. 8–16), 224px, BERT max_len 256,
set PYTORCH_ENABLE_MPS_FALLBACK=1. Encoders can be (un)frozen via `freeze_encoders`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models as tvm
    from transformers import AutoModel
    _TORCH = True
except Exception:
    _TORCH = False
    nn = object  # type: ignore


def pick_device() -> str:
    if not _TORCH:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if _TORCH:

    class _ImageEncoder(nn.Module):
        def __init__(self, embed_dim: int, pretrained: bool = True):
            super().__init__()
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = tvm.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(net.children())[:-1])  # drop fc
            self.proj = nn.Linear(2048, embed_dim)

        def forward(self, x):
            h = self.backbone(x).flatten(1)
            return self.proj(h)

    class _TextEncoder(nn.Module):
        def __init__(self, embed_dim: int, model_name: str = "bert-base-uncased"):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]      # [CLS]
            return self.proj(cls)

    class MultimodalNet(nn.Module):
        def __init__(self, embed_dim: int, num_classes: int,
                     text_model: str = "bert-base-uncased",
                     pretrained: bool = True, freeze_text: bool = False,
                     use_min: bool = False, proto_dim: int = 0):
            super().__init__()
            self.img = _ImageEncoder(embed_dim, pretrained)
            self.txt = _TextEncoder(embed_dim, text_model)
            # FedProto: insert a SHARED-dimension projection (proto space)
            # between fusion and head, so prototypes from clients with
            # different embed_dim live in one comparable space. The classifier
            # CONSUMES the proto space (encode -> proto -> head), so BCE
            # gradients flow through it — that is what prevents the demo-era
            # collapse (a proto head trained only by the pull loss can shrink
            # everything to a single point). proto_dim=0 -> plain head.
            self.proto_dim = proto_dim
            if proto_dim and proto_dim > 0:
                self.proto = nn.Linear(embed_dim, proto_dim)
                self.head = nn.Linear(proto_dim, num_classes)
            else:
                self.proto = None
                self.head = nn.Linear(embed_dim, num_classes)
            self.embed_dim = embed_dim
            self.freeze_text = freeze_text
            # MIN (Modality Imputation Network): predicts a TEXT embedding from the
            # IMAGE embedding, so a unimodal (image-only) client can synthesise a
            # pseudo-text embedding and become pseudo-multimodal (Saha's idea, at
            # the FEATURE level — no report-text generation needed because fusion
            # consumes embeddings, not words). Pre-trained on a multimodal client
            # BEFORE federation; used only when text is genuinely missing.
            self.use_min = use_min
            if use_min:
                self.min_net = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim))
            else:
                self.min_net = None
            if freeze_text:
                # freeze the whole BERT body (the text Linear proj stays trainable
                # so text embeddings can still adapt cheaply without BERT backprop)
                for p in self.txt.bert.parameters():
                    p.requires_grad = False

        def _text_embed(self, batch):
            """Return text embedding. If a precomputed BERT [CLS] is supplied
            (batch['text_feat']), skip BERT entirely and only apply the small
            trainable projection — this is the cache fast-path used when BERT is
            frozen, so BERT never runs during federated rounds."""
            if "text_feat" in batch:
                return self.txt.proj(batch["text_feat"])
            return self.txt(batch["input_ids"], batch["attention_mask"])

        def encode(self, batch) -> "torch.Tensor":
            return self.encode_mode(batch, "auto")

        def encode_mode(self, batch, mode: str = "auto") -> "torch.Tensor":
            """Mean-fuse embeddings over modalities ACTUALLY present in the batch.
            Uses the has_image/has_text flags (set by the dataset) rather than mere
            key presence, because unimodal clients still carry zero-filled text
            tensors. A client has fixed modalities, so the whole batch shares the
            same flag; we read the first element to decide."""
            def _present(flag_key):
                f = batch.get(flag_key)
                if f is None:
                    return True  # default: assume present (back-compat)
                if hasattr(f, "numel"):
                    return bool(f.reshape(-1)[0].item()) if f.numel() else False
                return bool(f)

            allowed = {"auto", "image_only", "text_only", "true_text", "min_text", "zero_text"}
            if mode not in allowed:
                raise ValueError(f"unknown modality mode {mode!r}")
            embs = []
            img_emb = None
            use_image = mode in {"image_only", "true_text", "min_text", "zero_text"}
            use_image = use_image or (mode == "auto" and _present("has_image"))
            if "image" in batch and use_image:
                img_emb = self.img(batch["image"])
                embs.append(img_emb)
            has_text_key = ("text_feat" in batch) or ("input_ids" in batch)
            use_real_text = mode in {"text_only", "true_text"}
            use_real_text = use_real_text or (
                mode == "auto" and has_text_key and _present("has_text")
            )
            if has_text_key and use_real_text:
                embs.append(self._text_embed(batch))
            elif (
                (mode == "min_text" or (mode == "auto" and self.use_min))
                and self.min_net is not None and img_emb is not None
            ):
                # text genuinely missing + MIN enabled: synthesise pseudo-text
                # embedding from the image embedding (pseudo-congruent MMFL)
                generated = self.min_net(img_emb)
                self.last_min_activation_norm = float(
                    generated.detach().norm(dim=1).mean().cpu()
                )
                embs.append(generated)
            elif mode == "zero_text" and img_emb is not None:
                embs.append(torch.zeros_like(img_emb))
            if not embs:
                raise ValueError("no modality present in batch")
            return torch.stack(embs, dim=0).mean(dim=0)

        def forward(self, batch, modality_mode: str = "auto"):
            z = self.encode_mode(batch, modality_mode)
            if self.proto is not None:
                z = self.proto(z)
            return self.head(z)


class TorchMultimodalBackend:
    """ModelBackend-compatible wrapper around MultimodalNet."""

    def __init__(self, ctx, dataset, embed_dim: int = 256,
                 share_encoders: bool = True, all_modalities: bool = False,
                 text_model: str = "bert-base-uncased", pretrained: bool = True,
                 freeze_encoders: bool = False, freeze_text: bool = False,
                 freeze_image: bool = False, use_min: bool = False,
                 proto_dim: int = 0, seed: int = 0):
        if not _TORCH:
            raise RuntimeError("torch/torchvision/transformers required")
        torch.manual_seed(seed + abs(getattr(ctx, "client_id", 0)))
        self.device = pick_device()
        self._ctx = ctx
        self._share = share_encoders
        self._num_classes = ctx.num_classes
        self.net = MultimodalNet(embed_dim, ctx.num_classes,
                                 text_model=text_model,
                                 pretrained=pretrained,
                                 freeze_text=freeze_text,
                                 use_min=use_min,
                                 proto_dim=proto_dim).to(self.device)
        if freeze_encoders:
            for p in self.net.img.backbone.parameters():
                p.requires_grad = False
            for p in self.net.txt.bert.parameters():
                p.requires_grad = False
        if freeze_image:
            # freeze ResNet backbone only (image proj stays trainable, like text).
            # Lets us A/B test "is training the image encoder worth its cost?"
            for p in self.net.img.backbone.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def precompute_text_features(self, loader):
        """Run BERT ONCE over a loader and return stacked [CLS] features (numpy).
        Used when BERT is frozen: cache these and feed as 'text_feat' so BERT never
        runs again during federated rounds (major speedup)."""
        self.net.eval()
        feats = []
        for batch in loader:
            b = self._to_device(batch)
            out = self.net.txt.bert(input_ids=b["input_ids"],
                                    attention_mask=b["attention_mask"])
            feats.append(out.last_hidden_state[:, 0].cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768))

    def pretrain_min(self, loader, epochs: int = 5, lr: float = 1e-3):
        """Pre-train the MIN (image-emb -> text-emb) on a MULTIMODAL loader,
        BEFORE federation. The MIN learns to reconstruct the real text embedding
        from the image embedding; later, unimodal clients use it to synthesise a
        pseudo-text embedding. Image/text encoders are frozen during this step so
        only the MIN MLP is fit. No effect unless use_min=True."""
        if not self.net.use_min or self.net.min_net is None:
            return {"gradient_norm": float("nan"), "update_norm": float("nan"),
                    "zero_gradient_rate": float("nan"), "loss": float("nan")}
        before = [parameter.detach().clone() for parameter in self.net.min_net.parameters()]
        self.net.train()
        opt = torch.optim.Adam(self.net.min_net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        gradient_sq = 0.0
        zero_gradients = gradient_elements = 0
        last = float("nan")
        for _ in range(epochs):
            for batch in loader:
                b = self._to_device(batch)
                with torch.no_grad():
                    img_emb = self.net.img(b["image"])
                    txt_emb = self.net._text_embed(b)
                pred = self.net.min_net(img_emb)
                loss = loss_fn(pred, txt_emb)
                opt.zero_grad()
                loss.backward()
                for parameter in self.net.min_net.parameters():
                    if parameter.grad is not None:
                        gradient_sq += float(parameter.grad.detach().pow(2).sum().cpu())
                        zero_gradients += int((parameter.grad.detach() == 0).sum().cpu())
                        gradient_elements += parameter.grad.numel()
                opt.step()
                last = float(loss.detach().cpu())
        update_sq = sum(
            float((parameter.detach() - initial).pow(2).sum().cpu())
            for parameter, initial in zip(self.net.min_net.parameters(), before)
        )
        return {
            "gradient_norm": float(gradient_sq ** 0.5),
            "update_norm": float(update_sq ** 0.5),
            "zero_gradient_rate": (
                zero_gradients / gradient_elements if gradient_elements else float("nan")
            ),
            "loss": last,
        }

    # ---- parameter exchange (numpy at the boundary, like the mock backend) ----
    def _shared_modules(self):
        """Which submodules are aggregated when share_encoders=False (head
        only). NOTE: the ①③④ runners all pass share_encoders=True, i.e. the
        FULL state_dict (minus min_net) is averaged; head-only is a legacy
        path kept for completeness."""
        if self._share:
            return self.net
        return self.net.head

    def shared_parameter_keys(self) -> List[str]:
        mod = self._shared_modules()
        prefix = "" if mod is self.net else "head."
        return [prefix + k for k in mod.state_dict().keys()]

    def get_parameters(self, only_shared: bool = True) -> Dict[str, np.ndarray]:
        if only_shared and not self._share:
            sd = self.net.head.state_dict()
            return {"head." + k: v.detach().cpu().numpy() for k, v in sd.items()}
        sd = self.net.state_dict()
        # MIN is a per-client pre-trained module (image->text imputation); it must
        # NOT be federated/averaged, or each client's imputation would be destroyed.
        return {k: v.detach().cpu().numpy() for k, v in sd.items()
                if not k.startswith("min_net.")}

    def set_parameters(self, params: Dict[str, np.ndarray],
                       only_shared: bool = True) -> None:
        sd = self.net.state_dict()
        for k, v in params.items():
            if k.startswith("min_net."):
                continue  # never overwrite the locally pre-trained MIN
            if k in sd:
                sd[k] = torch.tensor(v, device=self.device)
        self.net.load_state_dict(sd, strict=False)

    def get_min_parameters(self) -> Dict[str, np.ndarray]:
        if self.net.min_net is None:
            return {}
        return {
            key: value.detach().cpu().numpy()
            for key, value in self.net.min_net.state_dict().items()
        }

    def set_min_parameters(self, params: Dict[str, np.ndarray]) -> None:
        if self.net.min_net is None:
            if params:
                raise ValueError("backend has no MIN module")
            return
        state = self.net.min_net.state_dict()
        for key, value in params.items():
            if key in state:
                state[key] = torch.as_tensor(value, device=self.device)
        self.net.min_net.load_state_dict(state)

    # ---- batching helper ----
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if hasattr(v, "to") else v
        return out

    # ---- local training (multi-label BCE) ----
    def local_train(self, loader, epochs: int = 1, lr: float = 1e-4,
                    proximal_mu: float = 0.0, global_params=None,
                    extra=None) -> Dict[str, float]:
        self.net.train()
        min_parameters = (
            list(self.net.min_net.parameters()) if self.net.min_net is not None else []
        )
        min_before = [parameter.detach().clone() for parameter in min_parameters]
        min_gradient_sq = 0.0
        min_zero_gradients = min_gradient_elements = 0
        min_activation_norms = []
        opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad], lr=lr)
        last = 0.0
        for _ in range(max(epochs, 1)):
            for batch in loader:
                b = self._to_device(batch)
                opt.zero_grad()
                logits = self.net(b)
                loss = F.binary_cross_entropy_with_logits(logits, b["label"])
                if proximal_mu and global_params:
                    reg = 0.0
                    sd = self.net.state_dict()
                    for k, gv in global_params.items():
                        if k in sd:
                            reg = reg + ((sd[k] -
                                          torch.tensor(gv, device=self.device))
                                         ** 2).sum()
                    loss = loss + 0.5 * proximal_mu * reg
                loss.backward()
                for parameter in min_parameters:
                    if parameter.grad is not None:
                        min_gradient_sq += float(parameter.grad.detach().pow(2).sum().cpu())
                        min_zero_gradients += int((parameter.grad.detach() == 0).sum().cpu())
                        min_gradient_elements += parameter.grad.numel()
                activation = getattr(self.net, "last_min_activation_norm", None)
                if activation is not None:
                    min_activation_norms.append(float(activation))
                opt.step()
                last = float(loss.detach().cpu())
        metrics = {"loss": last}
        if min_parameters:
            update_sq = sum(
                float((parameter.detach() - initial).pow(2).sum().cpu())
                for parameter, initial in zip(min_parameters, min_before)
            )
            metrics.update({
                "min_gradient_norm": float(min_gradient_sq ** 0.5),
                "min_update_norm": float(update_sq ** 0.5),
                "min_zero_gradient_rate": (
                    min_zero_gradients / min_gradient_elements
                    if min_gradient_elements else float("nan")
                ),
                "min_activation_norm": (
                    float(np.mean(min_activation_norms))
                    if min_activation_norms else float("nan")
                ),
            })
        return metrics

    # ---- FedProto: proto-space training & per-label prototype statistics ----
    def local_train_proto(self, loader, epochs: int = 1, lr: float = 1e-4,
                          mu: float = 0.0,
                          global_protos=None) -> Dict[str, float]:
        """BCE (through the proto space) + mu * pull toward global per-label
        prototypes. The pull distance is averaged over proto dims (scale-
        stable) and over POSITIVE (sample, label) pairs only — multilabel-
        native. mu=0 (warmup) still trains proto+head via BCE; only the pull
        is off. global_protos: (num_classes, proto_dim) numpy or None."""
        assert self.net.proto is not None, "backend built without proto_dim"
        self.net.train()
        opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad], lr=lr)
        P = (torch.tensor(global_protos, device=self.device,
                          dtype=torch.float32)
             if global_protos is not None else None)
        last = 0.0
        for _ in range(max(epochs, 1)):
            for batch in loader:
                b = self._to_device(batch)
                z = self.net.proto(self.net.encode(b))        # (B, proto_dim)
                logits = self.net.head(z)
                loss = F.binary_cross_entropy_with_logits(logits, b["label"])
                if mu and P is not None:
                    y = b["label"]                            # (B, L)
                    d = ((z.unsqueeze(1) - P.unsqueeze(0)) ** 2).mean(-1)
                    loss = loss + mu * (d * y).sum() / y.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.detach().cpu())
        return {"loss": last}

    @torch.no_grad()
    def label_prototype_stats(self, loader):
        """Per-label POSITIVE prototype statistics in the shared proto space.
        Returns (S, C): S[l] = sum of proto embeddings over samples where
        label l is positive, C[l] = positive count. The server sums S and C
        across clients and sets P = S / C (positive-count-weighted mean).
        A sample contributes to EVERY one of its positive labels."""
        assert self.net.proto is not None
        self.net.eval()
        D = self.net.proto.out_features
        L = self._num_classes
        S = np.zeros((L, D), dtype=np.float64)
        C = np.zeros(L, dtype=np.float64)
        for batch in loader:
            b = self._to_device(batch)
            z = self.net.proto(self.net.encode(b)).cpu().numpy()   # (B, D)
            y = b["label"].cpu().numpy()                           # (B, L)
            S += y.T @ z
            C += y.sum(axis=0)
        return S, C

    @torch.no_grad()
    def predict_logits(self, loader) -> np.ndarray:
        self.net.eval()
        outs = []
        for batch in loader:
            b = self._to_device(batch)
            outs.append(self.net(b).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0,
                                                       self._num_classes))

    @torch.no_grad()
    def embed(self, loader) -> np.ndarray:
        self.net.eval()
        outs = []
        for batch in loader:
            b = self._to_device(batch)
            outs.append(self.net.encode(b).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0,
                                                       self.net.embed_dim))

    def distill(self, loader, soft_targets: np.ndarray, epochs: int = 1,
                lr: float = 1e-4, temperature: float = 2.0) -> float:
        """FedMD distillation: match consensus logits on public data (multi-label
        -> per-class sigmoid soft targets, BCE on tempered probabilities)."""
        self.net.train()
        opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad], lr=lr)
        T = temperature
        tgt = torch.tensor(soft_targets, device=self.device, dtype=torch.float32)
        last = 0.0
        ptr = 0
        for _ in range(max(epochs, 1)):
            for batch in loader:
                b = self._to_device(batch)
                bs = b["label"].shape[0]
                t = torch.sigmoid(tgt[ptr:ptr + bs] / T)
                ptr += bs
                opt.zero_grad()
                logits = self.net(b)
                loss = F.binary_cross_entropy_with_logits(logits / T, t)
                loss.backward()
                opt.step()
                last = float(loss.detach().cpu())
            ptr = 0
        return last

    def align_embeddings(self, loader, target_z: np.ndarray,
                         epochs: int = 1, lr: float = 1e-4) -> float:
        """LOOT: pull this model's embeddings toward target (leave-one-out mean)."""
        self.net.train()
        opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad], lr=lr)
        tgt = torch.tensor(target_z, device=self.device, dtype=torch.float32)
        last = 0.0
        ptr = 0
        for _ in range(max(epochs, 1)):
            for batch in loader:
                b = self._to_device(batch)
                bs = b["label"].shape[0]
                z = self.net.encode(b)
                loss = F.mse_loss(z, tgt[ptr:ptr + bs])
                ptr += bs
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.detach().cpu())
            ptr = 0
        return last

    @torch.no_grad()
    def class_prototypes(self, loader) -> Dict[int, np.ndarray]:
        """Mean embedding per (primary) class — for FedProto-style methods."""
        self.net.eval()
        sums: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for batch in loader:
            b = self._to_device(batch)
            z = self.net.encode(b).cpu().numpy()
            y = b["label"].cpu().numpy()
            for i in range(len(z)):
                pos = np.where(y[i] > 0)[0]
                cls = int(pos[0]) if len(pos) else 0
                sums[cls] = sums.get(cls, 0) + z[i]
                counts[cls] = counts.get(cls, 0) + 1
        return {c: sums[c] / counts[c] for c in sums}

    def embedding_dim(self) -> int:
        return self.net.embed_dim

    # ---- evaluation (multi-label metrics) ----
    @torch.no_grad()
    def prediction_arrays(self, loader, *, modality_mode: str = "auto"):
        self.net.eval()
        all_logits, all_y = [], []
        for batch in loader:
            b = self._to_device(batch)
            all_logits.append(self.net(b, modality_mode=modality_mode).cpu().numpy())
            all_y.append(b["label"].cpu().numpy())
        if not all_logits:
            return np.zeros((0, self._num_classes)), np.zeros((0, self._num_classes))
        logits = np.concatenate(all_logits, axis=0)
        y = np.concatenate(all_y, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return y, probs

    @torch.no_grad()
    def evaluate(
        self, loader, *, thresholds=None, rare_labels=None,
        modality_mode: str = "auto",
    ) -> Dict[str, float]:
        from ..evaluation.metrics import multilabel_report

        y, probs = self.prediction_arrays(loader, modality_mode=modality_mode)
        if not len(y):
            return {"accuracy": float("nan"), "macro_f1": float("nan"),
                    "auroc": float("nan"), "auprc": float("nan")}
        return multilabel_report(
            y, probs, thresholds=thresholds, rare_labels=rare_labels,
        )["summary"]

    @torch.no_grad()
    def min_reconstruction_diagnostics(self, loader) -> dict:
        """Compare true and generated text embeddings on multimodal samples."""

        if self.net.min_net is None:
            raise ValueError("MIN diagnostics require use_min=True")
        true_rows, generated_rows = [], []
        self.net.eval()
        for batch in loader:
            b = self._to_device(batch)
            image = self.net.img(b["image"])
            true = self.net._text_embed(b)
            generated = self.net.min_net(image)
            true_rows.append(true.cpu().numpy())
            generated_rows.append(generated.cpu().numpy())
        true = np.concatenate(true_rows, axis=0)
        generated = np.concatenate(generated_rows, axis=0)
        eps = 1e-12
        cosine = np.sum(true * generated, axis=1) / (
            np.linalg.norm(true, axis=1) * np.linalg.norm(generated, axis=1) + eps
        )
        normalized_mse = np.mean((true - generated) ** 2, axis=1) / (
            np.mean(true ** 2, axis=1) + eps
        )
        norm_ratio = np.linalg.norm(generated, axis=1) / (
            np.linalg.norm(true, axis=1) + eps
        )
        return {
            "samples": int(len(true)),
            "cosine_similarity": float(np.mean(cosine)),
            "normalized_mse": float(np.mean(normalized_mse)),
            "embedding_norm_ratio": float(np.mean(norm_ratio)),
        }
