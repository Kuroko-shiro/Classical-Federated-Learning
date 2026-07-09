"""Scenario 4 (model heterogeneity x modality incongruity) - FedMD-based runner.

The hardest cell of the 2x2 matrix: clients differ in BOTH architecture
(per-client embed_dim) AND modality (some multimodal, some image-only).

Why no MIN here (theoretical necessity, not a shortcut):
  Under model heterogeneity the embedding dimension d_k differs per client. An
  embedding-space imputation network (MIN) would need a teacher signal that is
  simultaneously (i) meaningful (from a *trained* encoder) and (ii) dimension-
  consistent (in client k's own R^{d_k}). These cannot hold together: a trained
  teacher lives in another client's d_j space (ii fails), while client k's own
  text encoder is never trained in a unimodal client (i fails). The only way to
  satisfy both is to move the teacher to LOGIT space (C-dim, shared, from trained
  clients) -- but that is exactly FedMD logit distillation. Hence in scenario 4
  the only coherent formulation lives in logit space. This necessity stems from
  model heterogeneity itself and is independent of the fusion mechanism. (MIN is
  used in scenario 3, where all clients share embed_dim and MIN is well-defined.)

Methods (both FedMD-based; differ ONLY in how the per-client target is formed):
  fedmd      : consensus = mean of ALL clients' public logits (standard FedMD).
  fedmd_loot : leave-one-out teacher in LOGIT space -- each client distills toward
               the mean logits of the OTHER clients (exclude self).

Run on the M5 (after splits/iu_split.json exists):
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/iu_federated_s4.py ... --method fedmd      --mm-ratio 1:3 --alpha 0.1
    python scripts/iu_federated_s4.py ... --method fedmd_loot --mm-ratio 1:3 --alpha 0.1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qflbench.data.iu_xray_prep import build_manifest
from qflbench.data.iu_xray_torch import IUXrayDataset
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device


class _Ctx:
    def __init__(self, cid, num_classes, modalities, n_train):
        self.client_id = cid
        self.num_classes = num_classes
        self.modalities = modalities
        self.num_train = n_train


def make_loader(manifest, idx, tok, img_size, batch, train, modalities,
                img_cache=None, num_workers=0):
    ds = IUXrayDataset(manifest, idx, tok, img_size=img_size, train=train,
                       modalities=modalities, img_cache=img_cache)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=num_workers)


def parse_ratio(s, clients):
    a, b = s.split(":")
    q, n = int(a), int(b)
    if q + n != clients:
        raise ValueError(f"--mm-ratio {s} must sum to --clients {clients}")
    return q, n


def vary_embed_dims(base, clients):
    """Per-client architecture heterogeneity: cycle through a set of embed_dims.
    Same recipe as scenario 2, so 2 and 4 share the 'model heterogeneity' notion."""
    palette = [128, 256, 192, 320, 224, 288, 160, 352]
    return [palette[i % len(palette)] for i in range(clients)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", default="fedmd",
                    choices=["fedmd", "fedmd_loot"])
    ap.add_argument("--mm-ratio", default="1:3",
                    help="multimodal:unimodal client ratio (e.g. 1:3 or 3:1)")
    ap.add_argument("--split", default="splits/iu_split.json",
                    help="frozen split JSON from scripts/iu_make_split.py")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--train-subset", type=int, default=2670)
    ap.add_argument("--test-subset", type=int, default=300)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--distill-epochs", type=int, default=1)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256,
                    help="base embed_dim; per-client dims vary around it")
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    # CSV setup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "iu")
    os.makedirs(results_dir, exist_ok=True)
    ratio_tag = args.mm_ratio.replace(":", "-")
    csv_path = os.path.join(
        results_dir,
        f"s4_{args.method}_{ratio_tag}_alpha{args.alpha}_c{args.clients}_{ts}.csv")
    csv_fields = ["round", "hamming_acc", "macro_f1", "auroc", "auprc", "seconds",
                  "comm_mb", "scenario", "method", "mm_ratio", "alpha", "clients",
                  "train_subset", "rounds", "batch", "distill_temp"]
    _csv_f = open(csv_path, "w", newline="")
    _csv_w = csv.DictWriter(_csv_f, fieldnames=csv_fields)
    _csv_w.writeheader()

    def _log(r, m, secs, comm):
        _csv_w.writerow({"round": r, "hamming_acc": m.get("hamming_acc"),
                         "macro_f1": m.get("macro_f1"), "auroc": m.get("auroc"),
                         "auprc": m.get("auprc"), "seconds": round(secs, 1),
                         "comm_mb": comm, "scenario": 4, "method": args.method,
                         "mm_ratio": args.mm_ratio, "alpha": args.alpha,
                         "clients": args.clients, "train_subset": args.train_subset,
                         "rounds": args.rounds, "batch": args.batch,
                         "distill_temp": args.distill_temp})
        _csv_f.flush()

    print(f"device = {pick_device()}  scenario=4 method={args.method}")
    print(f"results CSV -> {csv_path}")
    t0 = time.time()

    manifest = build_manifest(args.reports, args.projections, args.images,
                              require_findings=True, require_frontal=False)

    # === LOAD FROZEN SPLIT (same file as s1/s2/s3 -> all scenarios identical) ===
    import json as _json
    with open(args.split) as _sf:
        _SPLIT = _json.load(_sf)
    if _SPLIT["meta"]["manifest_size"] != len(manifest):
        raise RuntimeError(f"manifest size {len(manifest)} != split file "
                           f"{_SPLIT['meta']['manifest_size']}; rebuild split")
    public_idx = _SPLIT["public"]
    test_idx_full = _SPLIT["test"]
    train_pool = _SPLIT["train_pool"]
    akey = str(args.alpha)
    if akey not in _SPLIT["by_alpha"]:
        raise RuntimeError(f"alpha {akey} not in split file "
                           f"(have {list(_SPLIT['by_alpha'])}); rebuild split")
    part = {int(c): v for c, v in _SPLIT["by_alpha"][akey].items()}
    test_idx = test_idx_full[:args.test_subset]
    print(f"[frozen split] public={len(public_idx)} train_pool={len(train_pool)} "
          f"test={len(test_idx_full)} (from {args.split})")
    print("client sizes:", {c: len(v) for c, v in part.items()})

    # model heterogeneity: per-client embed_dim
    embed_dims = vary_embed_dims(args.embed_dim, args.clients)
    # modality incongruity: q multimodal + n unimodal (image-only)
    q, n = parse_ratio(args.mm_ratio, args.clients)
    client_mods = [["image", "text"]] * q + [["image"]] * n
    print(f"embed_dims (model heterogeneity): {embed_dims}")
    print(f"modality: {q} multimodal + {n} unimodal (image-only)")

    tok = AutoTokenizer.from_pretrained(args.text_model)

    # load image cache (dict) ONCE — passing the path string directly would break
    # IUXrayDataset which expects a dict.
    img_cache = None
    if args.img_cache:
        import torch as _t
        print(f"loading image cache: {args.img_cache}")
        img_cache = _t.load(args.img_cache)
        print(f"  cached images: {len(img_cache)}")

    # test = full modality, fixed (same as all scenarios)
    test_loader = make_loader(manifest, test_idx, tok, args.img_size, args.batch,
                              False, ["image", "text"],
                              img_cache=img_cache, num_workers=args.num_workers)
    # public set for FedMD logit exchange (full modality)
    public_loader = make_loader(manifest, public_idx, tok, args.img_size,
                                args.batch, False, ["image", "text"],
                                img_cache=img_cache, num_workers=args.num_workers)

    # per-client backends: model heterogeneity (different embed_dim); all clients
    # carry BOTH encoders; unimodal clients simply never feed text (no MIN in s4).
    clients = []
    for cid in range(args.clients):
        ctx = _Ctx(cid, 14, client_mods[cid], len(part[cid]))
        backend = TorchMultimodalBackend(
            ctx, dataset=None, embed_dim=embed_dims[cid],
            share_encoders=True, all_modalities=True, use_min=False,
            text_model=args.text_model, pretrained=True, seed=cid)
        loader = make_loader(manifest, part[cid], tok, args.img_size, args.batch,
                             True, client_mods[cid],
                             img_cache=img_cache, num_workers=args.num_workers)
        clients.append((ctx, backend, loader))

    print(f"setup {time.time()-t0:.1f}s. starting {args.rounds} rounds...")
    cum_bytes = 0

    for r in range(args.rounds):
        tr = time.time()
        round_bytes = 0
        # 1) local train + collect public logits
        logits = []
        for ctx, backend, loader in clients:
            backend.local_train(loader, epochs=args.local_epochs, lr=args.lr)
            lg = backend.predict_logits(public_loader)        # [N_pub, C]
            logits.append(lg)
            round_bytes += lg.size * lg.itemsize              # UPLOAD
        L = np.stack(logits, axis=0)                          # [clients, N_pub, C]

        # 2) per-client distillation target (the ONLY difference between methods)
        if args.method == "fedmd":
            consensus = L.mean(axis=0)                        # all-client mean
            for ctx, backend, loader in clients:
                round_bytes += consensus.size * consensus.itemsize   # DOWNLOAD
                backend.distill(public_loader, consensus,
                                epochs=args.distill_epochs,
                                lr=args.lr, temperature=args.distill_temp)
        else:  # fedmd_loot : leave-one-out mean in LOGIT space
            for ci, (ctx, backend, loader) in enumerate(clients):
                teacher = np.delete(L, ci, axis=0).mean(axis=0)  # exclude self
                round_bytes += teacher.size * teacher.itemsize   # DOWNLOAD
                backend.distill(public_loader, teacher,
                                epochs=args.distill_epochs,
                                lr=args.lr, temperature=args.distill_temp)

        cum_bytes += round_bytes
        comm_mb = cum_bytes / 1e6

        # 3) evaluate each client on the shared full-modality test, report mean
        ms = [b.evaluate(test_loader) for _, b, _ in clients]
        mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        secs = time.time() - tr
        print(f"[round {r:02d}] (mean) hamming={mean['hamming_acc']:.3f} "
              f"macro_f1={mean['macro_f1']:.3f} auroc={mean['auroc']:.3f} "
              f"auprc={mean['auprc']:.3f} | {secs:.1f}s | comm={comm_mb:.2f}MB")
        _log(r, mean, secs, comm_mb)

    print(f"\ntotal {time.time()-t0:.1f}s")
    _csv_f.close()
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
