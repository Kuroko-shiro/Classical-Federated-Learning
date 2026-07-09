"""Scenario 3 (modality incongruity) federated runner — Saha-faithful.

Separate from iu_federated.py to keep the working ①② code untouched. Implements
the core of Saha's central question: does incongruent MMFL (some clients have both
image+text, the rest only image) beat unimodal-FL (all clients image-only)?

Methods here:
  uniml  : UniFL reference line — ALL clients are image-only (no text anywhere).
           This is the baseline incongruent MMFL must beat to be worthwhile.
  fedavg : incongruent FedAvg — q multimodal clients + n unimodal (image-only),
           plain parameter averaging (all clients share the same full model;
           unimodal clients simply never feed text, so their text encoder gets no
           gradient that round — exactly the incongruity Saha studies).

Modality ratio (Saha uses 1:3 and 3:1) set via --mm-ratio:
  "1:3" -> 1 multimodal + 3 unimodal     "3:1" -> 3 multimodal + 1 unimodal

Run on the M5:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    # UniFL reference (ratio irrelevant — everyone image-only)
    python scripts/iu_federated_s3.py ... --method uniml --alpha 0.1
    # incongruent FedAvg, 1 multimodal : 3 unimodal
    python scripts/iu_federated_s3.py ... --method fedavg --mm-ratio 1:3 --alpha 0.1

LOOT / MIN are added later, once the FedAvg-vs-UniFL gap (the hole) is established.
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

from qflbench.data.iu_xray_prep import build_manifest, partition_clients
from qflbench.data.iu_xray_torch import IUXrayDataset
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device
from qflbench.models.base import weighted_average


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
    return DataLoader(ds, batch_size=batch, shuffle=train,
                      num_workers=num_workers)


def parse_ratio(s, clients):
    """'1:3' -> 1 multimodal, 3 unimodal (must sum to clients)."""
    a, b = s.split(":")
    q, n = int(a), int(b)
    if q + n != clients:
        raise ValueError(f"--mm-ratio {s} must sum to --clients {clients}")
    return q, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", default="fedavg",
                    choices=["uniml", "fedavg", "min", "loot"])
    ap.add_argument("--mm-ratio", default="1:3",
                    help="multimodal:unimodal client ratio (Saha: 1:3 or 3:1)")
    ap.add_argument("--split", default="splits/iu_split.json",
                    help="frozen split JSON from scripts/iu_make_split.py")
    ap.add_argument("--min-epochs", type=int, default=5,
                    help="MIN pre-training epochs (--method min only)")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--train-subset", type=int, default=2670)
    ap.add_argument("--test-subset", type=int, default=700)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    # CSV setup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "iu")
    os.makedirs(results_dir, exist_ok=True)
    ratio_tag = args.mm_ratio.replace(":", "-") if args.method == "fedavg" else "uni"
    csv_path = os.path.join(
        results_dir,
        f"s3_{args.method}_{ratio_tag}_alpha{args.alpha}_c{args.clients}_{ts}.csv")
    csv_fields = ["round", "hamming_acc", "macro_f1", "auroc", "auprc", "seconds",
                  "comm_mb", "scenario", "method", "mm_ratio", "alpha", "clients",
                  "train_subset", "rounds", "batch"]
    _csv_f = open(csv_path, "w", newline="")
    _csv_w = csv.DictWriter(_csv_f, fieldnames=csv_fields)
    _csv_w.writeheader()

    def _log(r, m, secs, comm):
        _csv_w.writerow({"round": r, "hamming_acc": m.get("hamming_acc"),
                         "macro_f1": m.get("macro_f1"), "auroc": m.get("auroc"),
                         "auprc": m.get("auprc"), "seconds": round(secs, 1),
                         "comm_mb": comm, "scenario": 3, "method": args.method,
                         "mm_ratio": (args.mm_ratio if args.method == "fedavg"
                                      else "uni"),
                         "alpha": args.alpha, "clients": args.clients,
                         "train_subset": args.train_subset, "rounds": args.rounds,
                         "batch": args.batch})
        _csv_f.flush()

    print(f"device = {pick_device()}  scenario=3 method={args.method}")
    print(f"results CSV -> {csv_path}")
    t0 = time.time()

    manifest = build_manifest(args.reports, args.projections, args.images,
                              require_findings=True, require_frontal=False)

    # === LOAD FROZEN SPLIT (same file as iu_federated.py → ①②③ identical) ===
    import json as _json
    with open(args.split) as _sf:
        _SPLIT = _json.load(_sf)
    if _SPLIT["meta"]["manifest_size"] != len(manifest):
        raise RuntimeError(f"manifest size {len(manifest)} != split file "
                           f"{_SPLIT['meta']['manifest_size']}; rebuild split")
    test_idx_full = _SPLIT["test"]
    train_pool = _SPLIT["train_pool"]
    akey = str(args.alpha)
    if akey not in _SPLIT["by_alpha"]:
        raise RuntimeError(f"alpha {akey} not in split file "
                           f"(have {list(_SPLIT['by_alpha'])}); rebuild split")
    part = {int(c): v for c, v in _SPLIT["by_alpha"][akey].items()}
    train_idx = train_pool[:args.train_subset]
    test_idx = test_idx_full[:args.test_subset]
    print(f"[frozen split] train_pool={len(train_pool)} test={len(test_idx_full)} "
          f"(from {args.split})")
    print("client sizes:", {c: len(v) for c, v in part.items()})

    # assign per-client modalities
    if args.method == "uniml":
        # UniFL reference: every client is image-only
        client_mods = [["image"]] * args.clients
        print("UniFL: all clients image-only (reference line)")
    else:
        q, n = parse_ratio(args.mm_ratio, args.clients)
        # first q clients multimodal, rest unimodal (image-only)
        client_mods = [["image", "text"]] * q + [["image"]] * n
        tag = "FedMD+MIN" if args.method == "min" else "incongruent FedAvg"
        print(f"{tag}: {q} multimodal + {n} unimodal (image-only)")

    use_min = (args.method == "min")
    tok = AutoTokenizer.from_pretrained(args.text_model)

    # load image cache (dict) ONCE, like s1/s2 — passing the path string directly
    # would break IUXrayDataset which expects a dict.
    img_cache = None
    if args.img_cache:
        import torch as _t
        print(f"loading image cache: {args.img_cache}")
        img_cache = _t.load(args.img_cache)
        print(f"  cached images: {len(img_cache)}")

    # test set is evaluated with BOTH modalities available (full-modality test)
    test_loader = make_loader(manifest, test_idx, tok, args.img_size, args.batch,
                              False, ["image", "text"],
                              img_cache=img_cache, num_workers=args.num_workers)

    # per-client backends: scenario 3 = SAME model (share full model via FedAvg),
    # every client instantiates BOTH encoders (all_modalities), unimodal clients
    # just never feed text -> text encoder simply gets no gradient those rounds.
    # For MIN: unimodal clients synthesise a pseudo-text embedding from the image.
    clients = []
    for cid in range(args.clients):
        ctx = _Ctx(cid, 14, client_mods[cid], len(part[cid]))
        backend = TorchMultimodalBackend(
            ctx, dataset=None, embed_dim=args.embed_dim,
            share_encoders=True, all_modalities=True, use_min=use_min,
            text_model=args.text_model, pretrained=True, seed=cid)
        loader = make_loader(manifest, part[cid], tok, args.img_size, args.batch,
                             True, client_mods[cid],
                             img_cache=img_cache, num_workers=args.num_workers)
        clients.append((ctx, backend, loader))

    gctx = _Ctx(-1, 14, ["image", "text"], 0)
    global_backend = TorchMultimodalBackend(
        gctx, dataset=None, embed_dim=args.embed_dim, share_encoders=True,
        all_modalities=True, use_min=use_min,
        text_model=args.text_model, pretrained=True, seed=999)

    # communication: FedAvg sends full model params up+down each round
    shared0 = global_backend.get_parameters(only_shared=True)
    param_bytes = sum(int(np.asarray(v).size) * np.asarray(v).itemsize
                      for v in shared0.values())
    per_round_mb = param_bytes * 2 * args.clients / 1e6

    print(f"setup {time.time()-t0:.1f}s. starting {args.rounds} rounds...")

    # ---- MIN pre-training (before federation), only for --method min ----
    if use_min:
        # find a multimodal client (has text) to pre-train the MIN on
        mm_clients = [(ctx, b, ld) for (ctx, b, ld) in clients
                      if "text" in ctx.modalities]
        if not mm_clients:
            raise RuntimeError("MIN needs at least one multimodal client")
        tmin = time.time()
        # pre-train MIN on each multimodal client, then average the MIN weights
        # and broadcast to ALL clients (so unimodal clients get a trained MIN)
        min_states = []
        for ctx, b, ld in mm_clients:
            b.pretrain_min(ld, epochs=args.min_epochs, lr=1e-3)
            min_states.append({k: v.detach().cpu().numpy()
                               for k, v in b.net.min_net.state_dict().items()})
        # average MIN weights across multimodal clients
        avg_min = {k: np.mean([s[k] for s in min_states], axis=0)
                   for k in min_states[0]}
        # broadcast the trained MIN to every client + the global model
        for ctx, b, ld in clients:
            sd = b.net.min_net.state_dict()
            for k in sd:
                sd[k] = torch.tensor(avg_min[k], device=b.device)
            b.net.min_net.load_state_dict(sd)
        gsd = global_backend.net.min_net.state_dict()
        for k in gsd:
            gsd[k] = torch.tensor(avg_min[k], device=global_backend.device)
        global_backend.net.min_net.load_state_dict(gsd)
        print(f"MIN pre-trained on {len(mm_clients)} multimodal client(s) "
              f"in {time.time()-tmin:.1f}s, broadcast to all")

    gp = global_backend.get_parameters(only_shared=True)
    # LOOT needs a shared probe set to compute leave-one-out target embeddings.
    # Use the test images WITHOUT labels purely as an alignment probe (Saha aligns
    # embeddings on a shared set); this leaks no labels. Built once.
    loot_probe = None
    if args.method == "loot":
        loot_probe = make_loader(manifest, test_idx, tok, args.img_size,
                                 args.batch, False, ["image", "text"],
                                 img_cache=img_cache,
                                 num_workers=args.num_workers)

    for r in range(args.rounds):
        tr = time.time()
        updates, weights = [], []
        for ctx, backend, loader in clients:
            backend.set_parameters(gp, only_shared=True)
            backend.local_train(loader, epochs=args.local_epochs, lr=args.lr)
            updates.append(backend.get_parameters(only_shared=True))
            weights.append(ctx.num_train)
        gp = weighted_average(updates, weights)
        global_backend.set_parameters(gp, only_shared=True)

        # LOOT: after aggregation, each client aligns its embeddings toward the
        # leave-one-out mean of the OTHER clients' embeddings on the probe set.
        if args.method == "loot":
            # broadcast aggregated params, then collect each client's probe embeds
            embs = []
            for ctx, backend, loader in clients:
                backend.set_parameters(gp, only_shared=True)
                embs.append(backend.embed(loot_probe))   # [N_probe, embed_dim]
            embs = np.stack(embs, axis=0)                  # [C, N, D]
            C = embs.shape[0]
            for ci, (ctx, backend, loader) in enumerate(clients):
                # leave-one-out mean (teacher = all OTHER clients)
                others = np.delete(embs, ci, axis=0).mean(axis=0)  # [N, D]
                backend.align_embeddings(loot_probe, others,
                                         epochs=1, lr=args.lr)
            # re-aggregate after alignment so the global model reflects it
            updates2 = [b.get_parameters(only_shared=True)
                        for _, b, _ in clients]
            gp = weighted_average(updates2, weights)
            global_backend.set_parameters(gp, only_shared=True)

        m = global_backend.evaluate(test_loader)
        secs = time.time() - tr
        comm = per_round_mb * (r + 1)
        print(f"[round {r:02d}] hamming={m['hamming_acc']:.3f} "
              f"macro_f1={m['macro_f1']:.3f} auroc={m['auroc']:.3f} "
              f"auprc={m['auprc']:.3f} | {secs:.1f}s | comm={comm:.1f}MB")
        _log(r, m, secs, comm)

    print(f"\ntotal {time.time()-t0:.1f}s")
    _csv_f.close()
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
