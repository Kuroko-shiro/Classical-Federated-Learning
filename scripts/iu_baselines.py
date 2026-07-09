"""
Baseline runner: Centralized (upper bound) and Local-only (lower bound).

Why: FL results alone don't tell whether federation helped. We need
  gap_filled = (FL - Local) / (Centralized - Local)
to quantify "how much of the local->centralized gap did federation close".
This framework existed in the synthetic-data phase (smoke_test.py) but was
never ported to the real IU X-ray runners. This is that port.

- Centralized: train ONE model on the full train_pool subset (no federation,
  single client holding all data). Alpha-independent. Upper bound / skyline.
- Local-only: each client trains on its OWN data only (the SAME per-alpha,
  per-ratio partition the FL runs used), evaluated on the shared test, then
  averaged. No aggregation at all. Lower bound / floor.

Reuses everything from the FL runners: same manifest, same frozen split, same
backend, same local_train, same evaluate, same metrics. Only the coordination
is removed, so the numbers are directly comparable to FL.

Heterogeneity handling (to match each scenario's floor/ceiling):
- --vary-embed     : model heterogeneity (embed_dim per client {128,256,192,320}).
                     Use for the LOCAL baseline of scenarios ② and ④.
- --mm-ratio q:n   : modality incongruity (q multimodal + n image-only clients).
                     Use for the LOCAL baseline of scenarios ③ and ④.
Centralized is a single full-modality model (the achievable ceiling if all data
and both modalities could be pooled); for a model-het ceiling reference, run
centralized at a representative embed_dim (default 256).
"""
import argparse
import os
import time
import json as _json
import csv
from datetime import datetime

import sys

import numpy as np
from transformers import AutoTokenizer

# make qflbench importable (same as the FL runners)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qflbench.data.iu_xray_prep import build_manifest
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device
from qflbench.metrics.classification import multilabel_metrics  # noqa: F401
from qflbench.data.iu_xray_torch import IUXrayDataset
from torch.utils.data import DataLoader


class _Ctx:
    def __init__(self, cid, num_classes, modalities, n_train):
        self.client_id = cid
        self.num_classes = num_classes
        self.modalities = modalities
        self.n_train = n_train


def make_loader(manifest, idx, tok, img_size, batch, train, modalities,
                img_cache=None, num_workers=0):
    ds = IUXrayDataset(manifest, idx, tok, img_size=img_size, train=train,
                       modalities=modalities, img_cache=img_cache)
    return DataLoader(ds, batch_size=batch, shuffle=train,
                      num_workers=num_workers)


def vary_embed_dims(base, clients):
    palette = [128, 256, 192, 320, 224, 288, 160, 352]
    return [palette[i % len(palette)] for i in range(clients)]


def client_modalities(n_clients, mm_ratio):
    """q multimodal + n unimodal(image-only), matching s3/s4 convention."""
    if mm_ratio is None:
        return [["image", "text"]] * n_clients
    q, n = [int(x) for x in mm_ratio.split(":")]
    # scale to n_clients (same logic family as s3/s4: first q multimodal)
    total = q + n
    reps = max(1, n_clients // total)
    mods = ([["image", "text"]] * q + [["image"]] * n) * reps
    mods = mods[:n_clients]
    while len(mods) < n_clients:
        mods.append(["image"])
    return mods


def lr_at(base_lr, epoch, total_epochs, schedule):
    """Learning rate for a given epoch."""
    if schedule == "cosine":
        import math
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))
    return base_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--mode", required=True, choices=["centralized", "local"],
                    help="centralized=upper bound (all data, 1 model); "
                         "local=lower bound (each client its own data, averaged)")
    ap.add_argument("--split", default="splits/iu_split.json")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=100.0,
                    help="for local: which per-alpha partition to use")
    ap.add_argument("--epochs", type=int, default=40,
                    help="training epochs (match FL total exposure: rounds*local_epochs)")
    ap.add_argument("--train-subset", type=int, default=2670)
    ap.add_argument("--test-subset", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-decay", choices=["none", "cosine"], default="none",
                    help="cosine: decay lr from --lr to ~0 over epochs "
                         "(fixes late-epoch instability for centralized)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--vary-embed", action="store_true",
                    help="model heterogeneity: vary embed_dim per client "
                         "(for LOCAL baseline of scenarios ②④)")
    ap.add_argument("--mm-ratio", default=None,
                    help="modality incongruity q:n, e.g. 1:3 or 3:1 "
                         "(for LOCAL baseline of scenarios ③④)")
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2,
                    help="evaluate every N epochs (cheaper; default 2)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "iu")
    os.makedirs(results_dir, exist_ok=True)
    tag = args.mode
    if args.mode == "local":
        tag += f"_a{args.alpha}"
        if args.mm_ratio:
            tag += f"_{args.mm_ratio.replace(':','-')}"
        if args.vary_embed:
            tag += "_hetero"
    csv_path = os.path.join(results_dir,
                            f"baseline_{tag}_c{args.clients}_{ts}.csv")
    csv_fields = ["epoch", "hamming_acc", "macro_f1", "auroc", "auprc",
                  "seconds", "mode", "alpha", "clients", "mm_ratio",
                  "vary_embed", "which_client"]
    _csv_f = open(csv_path, "w", newline="")
    _csv_w = csv.DictWriter(_csv_f, fieldnames=csv_fields)
    _csv_w.writeheader()

    def log(epoch, m, secs, which="all"):
        _csv_w.writerow({
            "epoch": epoch, "hamming_acc": m.get("hamming_acc"),
            "macro_f1": m.get("macro_f1"), "auroc": m.get("auroc"),
            "auprc": m.get("auprc"), "seconds": round(secs, 1),
            "mode": args.mode, "alpha": args.alpha, "clients": args.clients,
            "mm_ratio": args.mm_ratio or "full", "vary_embed": args.vary_embed,
            "which_client": which})
        _csv_f.flush()

    print(f"results CSV -> {csv_path}")
    print(f"device = {pick_device()}  mode={args.mode}")
    t0 = time.time()

    manifest = build_manifest(args.reports, args.projections, args.images,
                              require_findings=True, require_frontal=False)
    with open(args.split) as sf:
        SP = _json.load(sf)
    if SP["meta"]["manifest_size"] != len(manifest):
        raise RuntimeError(f"manifest {len(manifest)} != split "
                           f"{SP['meta']['manifest_size']}; rebuild split")
    train_pool = SP["train_pool"]
    test_idx = SP["test"][:args.test_subset]
    train_idx_full = train_pool[:args.train_subset]

    img_cache = None
    if args.img_cache:
        import torch as _t
        print(f"loading image cache: {args.img_cache}")
        img_cache = _t.load(args.img_cache)
        print(f"  cached images: {len(img_cache)}")

    tok = AutoTokenizer.from_pretrained(args.text_model)
    # test is ALWAYS full modality (same as all FL runs)
    test_loader = make_loader(manifest, test_idx, tok, args.img_size, args.batch,
                              False, ["image", "text"],
                              img_cache=img_cache, num_workers=args.num_workers)

    if args.mode == "centralized":
        # ---- UPPER BOUND: one model, all train_pool data, full modality ----
        print(f"[centralized] training 1 model on {len(train_idx_full)} samples "
              f"(embed_dim={args.embed_dim}, full modality)")
        ctx = _Ctx(0, 14, ["image", "text"], len(train_idx_full))
        backend = TorchMultimodalBackend(
            ctx, dataset=None, embed_dim=args.embed_dim, share_encoders=True,
            text_model=args.text_model, pretrained=True, seed=0)
        loader = make_loader(manifest, train_idx_full, tok, args.img_size,
                             args.batch, True, ["image", "text"],
                             img_cache=img_cache, num_workers=args.num_workers)
        print(f"setup {time.time()-t0:.1f}s. training {args.epochs} epochs "
              f"(lr_decay={args.lr_decay})...")
        for ep in range(args.epochs):
            te = time.time()
            cur_lr = lr_at(args.lr, ep, args.epochs, args.lr_decay)
            backend.local_train(loader, epochs=1, lr=cur_lr)
            if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                m = backend.evaluate(test_loader)
                log(ep, m, time.time() - te)
                print(f"[ep {ep:02d}] hamming={m.get('hamming_acc'):.3f} "
                      f"macro_f1={m.get('macro_f1'):.3f} "
                      f"auroc={m.get('auroc'):.3f} | lr={cur_lr:.2e} | "
                      f"{time.time()-te:.1f}s")
        print(f"\ntotal {time.time()-t0:.1f}s -> {csv_path}")

    else:
        # ---- LOWER BOUND: each client trains on ITS OWN data, then average ----
        akey = str(args.alpha)
        if akey not in SP["by_alpha"]:
            raise RuntimeError(f"alpha {akey} not in split; have "
                               f"{list(SP['by_alpha'])}")
        part = {int(c): v for c, v in SP["by_alpha"][akey].items()}
        # cap each client's data to the same pool the FL runs used
        pool_set = set(train_idx_full)
        part = {c: [i for i in idx if i in pool_set] for c, idx in part.items()}

        embed_dims = (vary_embed_dims(args.embed_dim, args.clients)
                      if args.vary_embed else [args.embed_dim] * args.clients)
        mods = client_modalities(args.clients, args.mm_ratio)
        print(f"[local] {args.clients} clients train in ISOLATION then averaged")
        print(f"  embed_dims={embed_dims}")
        print(f"  modalities={mods}")
        print(f"  client sizes={{{', '.join(f'{c}:{len(v)}' for c,v in part.items())}}}")

        final_per_client = []
        peak_per_client = []
        for cid in range(args.clients):
            ctx = _Ctx(cid, 14, mods[cid], len(part[cid]))
            backend = TorchMultimodalBackend(
                ctx, dataset=None, embed_dim=embed_dims[cid], share_encoders=True,
                text_model=args.text_model, pretrained=True, seed=cid)
            loader = make_loader(manifest, part[cid], tok, args.img_size,
                                 args.batch, True, mods[cid],
                                 img_cache=img_cache, num_workers=args.num_workers)
            print(f"  -- client {cid}: {len(part[cid])} samples, "
                  f"embed={embed_dims[cid]}, mods={mods[cid]}")
            client_auroc_hist = []
            best_m = None
            for ep in range(args.epochs):
                cur_lr = lr_at(args.lr, ep, args.epochs, args.lr_decay)
                backend.local_train(loader, epochs=1, lr=cur_lr)
                if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                    m = backend.evaluate(test_loader)
                    client_auroc_hist.append(m.get("auroc") or 0.0)
                    if best_m is None or (m.get("auroc") or 0) > (best_m.get("auroc") or 0):
                        best_m = m
                    log(ep, m, time.time() - t0, which=f"client{cid}")
            m = best_m  # use this client's PEAK (robust to late instability)
            print(f"     client {cid}: peak auroc={m.get('auroc'):.3f} "
                  f"macro_f1={m.get('macro_f1'):.3f}  "
                  f"(final={client_auroc_hist[-1]:.3f}, "
                  f"{'stable' if client_auroc_hist[-1] >= m.get('auroc')-0.03 else 'LATE-DROP'})")
            final_per_client.append(m)

        # average across clients = the Local lower bound (using per-client peaks)
        avg = {}
        for k in ["hamming_acc", "macro_f1", "auroc", "auprc"]:
            vals = [m.get(k) for m in final_per_client if m.get(k) is not None]
            avg[k] = float(np.mean(vals)) if vals else None
        log(args.epochs - 1, avg, time.time() - t0, which="AVERAGE")
        print(f"\n[LOCAL average] auroc={avg['auroc']:.3f} "
              f"macro_f1={avg['macro_f1']:.3f}  (lower bound, per-client peaks)")
        print(f"total {time.time()-t0:.1f}s -> {csv_path}")

    _csv_f.close()


if __name__ == "__main__":
    main()
