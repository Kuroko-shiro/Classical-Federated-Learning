"""Federated runner for IU X-ray real data (method B).

Separate from the synthetic-data simulator so we don't disturb the working ①②③
harness. It wires: manifest -> per-client DataLoaders (Dirichlet non-IID) ->
TorchMultimodalBackend per client -> a federated strategy (FedAvg/FedProx/FedMD).

Scenario ① here = same model, same modality (both image+text everywhere).
Scenario ② = different model (we vary embed_dim per client) , same modality.
(③④ add modality masking; layered once ①② run on real data.)

Run on the M5:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/iu_federated.py \
        --reports data/indiana_reports.csv \
        --projections data/indiana_projections.csv \
        --images data/images/images_normalized \
        --scenario 1 --method fedavg \
        --clients 2 --rounds 10 --alpha 0.5 \
        --train-subset 800 --local-epochs 2 --batch 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qflbench.data.iu_xray_prep import (build_manifest, split_train_test,
                                        partition_clients)
from qflbench.data.iu_xray_torch import IUXrayDataset
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device
from qflbench.models.base import weighted_average, hetero_aggregate, slice_to
from qflbench.metrics.classification import multilabel_metrics


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


def evaluate_global(backend, test_loader):
    return backend.evaluate(test_loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    ap.add_argument("--split", default="splits/iu_split.json",
                    help="frozen split JSON from scripts/iu_make_split.py")
    ap.add_argument("--method", default="fedavg",
                    choices=["fedavg", "fedprox", "fedmd",
                             "heterofl", "fedproto"])
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--all-data", action="store_true",
                    help="give every client the UNION of all split partitions. "
                         "Use with --clients 1 --method fedavg to run the "
                         "Centralized ceiling through the exact FL harness/"
                         "schedule (data identical to the 4-client union).")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--train-subset", type=int, default=800)
    ap.add_argument("--test-subset", type=int, default=300)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--distill-epochs", type=int, default=1,
                    help="FedMD: distillation epochs per round (default 1)")
    ap.add_argument("--distill-temp", type=float, default=2.0,
                    help="FedMD: distillation temperature (default 2.0; "
                         "lower=sharper targets)")
    ap.add_argument("--distill-lr", type=float, default=None,
                    help="FedMD: distillation LR (default: same as --lr)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mu", type=float, default=0.1, help="FedProx proximal mu")
    ap.add_argument("--proto-dim", type=int, default=128,
                    help="FedProto: shared prototype-space dim (per-client "
                         "Linear(embed_dim_k -> proto_dim) feeds the head)")
    ap.add_argument("--proto-mu", type=float, default=0.1,
                    help="FedProto: pull strength toward global prototypes "
                         "after warmup (demo-era lesson: 1.0 collapses)")
    ap.add_argument("--proto-warmup", type=int, default=3,
                    help="FedProto: rounds with mu=0 before the pull starts")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--img-cache", default=None,
                    help="path to pre-resized image cache (.pt) from "
                         "scripts/iu_cache_images.py. Lossless speedup.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (parallel image loading). Try 2-4.")
    ap.add_argument("--freeze-text", action="store_true",
                    help="freeze BERT (image encoder still trains). Big speedup; "
                         "drops BERT backprop. Text proj stays trainable.")
    ap.add_argument("--freeze-image", action="store_true",
                    help="freeze ResNet backbone (image proj still trains). "
                         "Combined with --freeze-text => only fusion+head+projs "
                         "train (fastest). Use to A/B test image-encoder value.")
    args = ap.parse_args()

    # --- result CSV setup (timestamped, never overwritten) ---
    import csv
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "iu")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(
        results_dir,
        f"s{args.scenario}_{args.method}_alpha{args.alpha}_c{args.clients}_{ts}.csv")
    csv_fields = ["round", "hamming_acc", "macro_f1", "auroc", "auprc",
                  "seconds", "comm_mb", "scenario", "method", "alpha", "clients",
                  "train_subset", "rounds", "batch", "freeze_text", "freeze_image"]
    _csv_f = open(csv_path, "w", newline="")
    _csv_w = csv.DictWriter(_csv_f, fieldnames=csv_fields)
    _csv_w.writeheader()

    def _log_round(r, m, secs):
        _csv_w.writerow({
            "round": r, "hamming_acc": m.get("hamming_acc"),
            "macro_f1": m.get("macro_f1"), "auroc": m.get("auroc"),
            "auprc": m.get("auprc"), "seconds": round(secs, 1),
            "comm_mb": m.get("_comm_mb"),
            "scenario": args.scenario, "method": args.method, "alpha": args.alpha,
            "clients": args.clients, "train_subset": args.train_subset,
            "rounds": args.rounds, "batch": args.batch,
            "freeze_text": args.freeze_text, "freeze_image": args.freeze_image,
        })
        _csv_f.flush()  # write incrementally so a crash keeps finished rounds
    print(f"results CSV -> {csv_path}")

    # --- provenance: dump full config + git hash next to the CSV ---
    import json as _pjson
    import subprocess as _sp
    try:
        _git = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        _git = "nogit"
    with open(csv_path + ".config.json", "w") as _cf:
        _pjson.dump({**vars(args), "git": _git}, _cf, indent=1)
    print(f"config JSON -> {csv_path}.config.json (git={_git})")

    print(f"device = {pick_device()}  scenario={args.scenario} method={args.method}")
    t0 = time.time()

    manifest = build_manifest(args.reports, args.projections, args.images,
                              require_findings=True, require_frontal=False)

    # === LOAD FROZEN SPLIT (root-cause fix: split never recomputed/drifts) ===
    # Every scenario/method loads the SAME splits/iu_split.json produced once by
    # scripts/iu_make_split.py. public/test/train_pool and the per-alpha client
    # partition all come from that file, so cross-scenario comparison is exact.
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
    if args.all_data:
        merged = sorted({i for v in part.values() for i in v})
        part = {c: merged for c in range(args.clients)}
        print(f"[all-data] every client gets the union: {len(merged)} samples")

    train_idx = train_pool[:args.train_subset]
    test_idx = test_idx_full[:args.test_subset]
    needs_public = (args.method in ("fedmd", "loot"))
    print(f"[frozen split] public={len(public_idx)} train_pool={len(train_pool)} "
          f"test={len(test_idx_full)} (from {args.split})")
    print(f"[this run] train={len(train_idx)}, test={len(test_idx)}, "
          f"method={args.method} uses_public={needs_public}")
    print("client sizes:", {c: len(v) for c, v in part.items()})

    img_cache = None
    if args.img_cache:
        import torch as _t
        print(f"loading image cache: {args.img_cache}")
        img_cache = _t.load(args.img_cache)
        print(f"  cached images: {len(img_cache)}")

    tok = AutoTokenizer.from_pretrained(args.text_model)
    test_loader = make_loader(manifest, test_idx, tok, args.img_size, args.batch,
                              False, ["image", "text"],
                              img_cache=img_cache, num_workers=args.num_workers)

    # scenario 2 = model heterogeneity: vary embed_dim per client
    embed_dims = ([args.embed_dim] * args.clients if args.scenario == 1
                  else [128, 256, 192, 320, 224, 256][:args.clients]
                       + [256] * max(0, args.clients - 6))

    # per-client backends (share_encoders True for ①②: same modality)
    clients = []
    for cid in range(args.clients):
        ctx = _Ctx(cid, 14, ["image", "text"], len(part[cid]))
        backend = TorchMultimodalBackend(
            ctx, dataset=None, embed_dim=embed_dims[cid],
            share_encoders=True, text_model=args.text_model, pretrained=True,
            freeze_text=args.freeze_text, freeze_image=args.freeze_image,
            proto_dim=(args.proto_dim if args.method == "fedproto" else 0),
            seed=cid)
        loader = make_loader(manifest, part[cid], tok, args.img_size, args.batch,
                             True, ["image", "text"],
                             img_cache=img_cache, num_workers=args.num_workers)
        clients.append((ctx, backend, loader))

    # global model:
    #   fedavg/fedprox -> same-dim global | heterofl -> MAX-dim global template
    #   fedmd/fedproto -> no parameter server model (nothing is averaged)
    if args.method in ("fedavg", "fedprox", "heterofl"):
        gdim = max(embed_dims) if args.method == "heterofl" else args.embed_dim
        gctx = _Ctx(-1, 14, ["image", "text"], 0)
        global_backend = TorchMultimodalBackend(
            gctx, dataset=None, embed_dim=gdim, share_encoders=True,
            text_model=args.text_model, pretrained=True,
            freeze_text=args.freeze_text, freeze_image=args.freeze_image,
            seed=999)
        if args.method == "heterofl":
            print(f"[heterofl] client dims={embed_dims} -> global template "
                  f"dim={gdim}")

    print(f"setup done in {time.time()-t0:.1f}s. starting {args.rounds} rounds...")

    # ---- FedAvg / FedProx loop ----
    if args.method in ("fedavg", "fedprox"):
        gp = global_backend.get_parameters(only_shared=True)
        for r in range(args.rounds):
            tr = time.time()
            updates, weights = [], []
            round_bytes = 0
            for ctx, backend, loader in clients:
                round_bytes += sum(v.nbytes for v in gp.values())  # download
                backend.set_parameters(gp, only_shared=True)
                backend.local_train(
                    loader, epochs=args.local_epochs, lr=args.lr,
                    proximal_mu=(args.mu if args.method == "fedprox" else 0.0),
                    global_params=(gp if args.method == "fedprox" else None))
                up = backend.get_parameters(only_shared=True)
                round_bytes += sum(v.nbytes for v in up.values())  # upload
                updates.append(up)
                weights.append(ctx.num_train)
            gp = weighted_average(updates, weights)
            main._cum_bytes = getattr(main, "_cum_bytes", 0) + round_bytes
            global_backend.set_parameters(gp, only_shared=True)
            m = evaluate_global(global_backend, test_loader)
            m["_comm_mb"] = main._cum_bytes / 1e6
            secs = time.time() - tr
            print(f"[round {r:02d}] hamming={m['hamming_acc']:.3f} "
                  f"macro_f1={m['macro_f1']:.3f} auroc={m['auroc']:.3f} "
                  f"auprc={m['auprc']:.3f} | {secs:.1f}s")
            _log_round(r, m, secs)
        print(f"\ntotal {time.time()-t0:.1f}s")
        _csv_f.close()
        print(f"saved: {csv_path}")
        return

    # ---- HeteroFL loop (nested width-sliced parameter averaging) ----
    # All clients are (re)initialised from the SAME global template slice at
    # round 0, so the shared coordinates are aligned by construction (this is
    # what FedProto lacked). Backbones (identical shapes) get a plain weighted
    # mean; width-varying layers get coverage-weighted nested averaging.
    if args.method == "heterofl":
        gp = global_backend.get_parameters(only_shared=True)  # max-dim
        for r in range(args.rounds):
            tr = time.time()
            round_bytes = 0
            updates, weights = [], []
            for ctx, backend, loader in clients:
                ref = backend.get_parameters(only_shared=True)
                sub = slice_to(gp, ref)
                round_bytes += sum(v.nbytes for v in sub.values())  # download
                backend.set_parameters(sub, only_shared=True)
                backend.local_train(loader, epochs=args.local_epochs,
                                    lr=args.lr)
                up = backend.get_parameters(only_shared=True)
                round_bytes += sum(v.nbytes for v in up.values())   # upload
                updates.append(up)
                weights.append(ctx.num_train)
            gp = hetero_aggregate(updates, weights)
            main._cum_bytes = getattr(main, "_cum_bytes", 0) + round_bytes
            # eval AFTER aggregation: each client rejoins at its own width;
            # report the per-client mean (comparable to FedMD / hetero-Local)
            ms = []
            for ctx, backend, loader in clients:
                ref = backend.get_parameters(only_shared=True)
                backend.set_parameters(slice_to(gp, ref), only_shared=True)
                ms.append(evaluate_global(backend, test_loader))
            mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            mean["_comm_mb"] = main._cum_bytes / 1e6
            secs = time.time() - tr
            print(f"[round {r:02d}] (HeteroFL mean) "
                  f"hamming={mean['hamming_acc']:.3f} "
                  f"macro_f1={mean['macro_f1']:.3f} "
                  f"auroc={mean['auroc']:.3f} "
                  f"| {secs:.1f}s | comm={mean['_comm_mb']:.1f}MB cumulative")
            _log_round(r, mean, secs)
        print(f"\ntotal {time.time()-t0:.1f}s")
        _csv_f.close()
        print(f"saved: {csv_path}")
        return

    # ---- FedProto loop (per-label prototypes in a shared proto space) ----
    # No model parameters cross the wire. Each round: clients train (BCE
    # through the proto space + warmed-up pull), then send per-label positive
    # sums/counts; server sets P = sum(S)/sum(C) (positive-count-weighted).
    # Demo-era lessons applied: shared 128-dim space via a learned projection
    # (coordinate alignment), classification stays on each LOCAL head, and
    # mu warms up from 0 instead of the paper's 1.0.
    if args.method == "fedproto":
        n_cls = 14
        global_protos = None
        for r in range(args.rounds):
            tr = time.time()
            mu = 0.0 if r < args.proto_warmup else args.proto_mu
            round_bytes = 0
            S_tot = np.zeros((n_cls, args.proto_dim), dtype=np.float64)
            C_tot = np.zeros(n_cls, dtype=np.float64)
            for ctx, backend, loader in clients:
                backend.local_train_proto(
                    loader, epochs=args.local_epochs, lr=args.lr,
                    mu=mu, global_protos=global_protos)
                S, C = backend.label_prototype_stats(loader)
                round_bytes += S.nbytes + C.nbytes                  # upload
                S_tot += S
                C_tot += C
            global_protos = (S_tot / np.maximum(C_tot, 1.0)[:, None]
                             ).astype(np.float32)
            round_bytes += global_protos.nbytes * len(clients)      # download
            main._cum_bytes = getattr(main, "_cum_bytes", 0) + round_bytes
            ms = [evaluate_global(b, test_loader) for _, b, _ in clients]
            mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            mean["_comm_mb"] = main._cum_bytes / 1e6
            secs = time.time() - tr
            print(f"[round {r:02d}] (FedProto mean, mu={mu:.2f}) "
                  f"hamming={mean['hamming_acc']:.3f} "
                  f"macro_f1={mean['macro_f1']:.3f} "
                  f"auroc={mean['auroc']:.3f} "
                  f"| {secs:.1f}s | comm={mean['_comm_mb']:.4f}MB cumulative")
            _log_round(r, mean, secs)
        print(f"\ntotal {time.time()-t0:.1f}s")
        _csv_f.close()
        print(f"saved: {csv_path}")
        return

    # ---- FedMD loop (logit distillation on a public subset) ----
    # public set was reserved up-front (held out from training); reuse it here
    public_loader = make_loader(manifest, public_idx, tok, args.img_size,
                                args.batch, False, ["image", "text"],
                                img_cache=img_cache, num_workers=args.num_workers)
    for r in range(args.rounds):
        tr = time.time()
        round_bytes = 0
        # 1) each client trains locally, then uploads its public-set logits
        logits_list = []
        for ctx, backend, loader in clients:
            backend.local_train(loader, epochs=args.local_epochs, lr=args.lr)
            lg = backend.predict_logits(public_loader)
            logits_list.append(lg)
            round_bytes += lg.size * lg.itemsize          # UPLOAD: client -> server
        # 2) consensus = mean logits
        consensus = np.mean(np.stack(logits_list, axis=0), axis=0)
        # 3) server broadcasts consensus to each client, who distills toward it
        for ctx, backend, loader in clients:
            round_bytes += consensus.size * consensus.itemsize  # DOWNLOAD: server -> client
            backend.distill(public_loader, consensus,
                            epochs=args.distill_epochs,
                            lr=(args.distill_lr if args.distill_lr else args.lr),
                            temperature=args.distill_temp)
        cum_comm_mb = (getattr(main, "_cum_bytes", 0) + round_bytes) / 1e6
        main._cum_bytes = getattr(main, "_cum_bytes", 0) + round_bytes
        # 4) eval each client, report mean
        ms = [evaluate_global(b, test_loader) for _, b, _ in clients]
        mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        secs = time.time() - tr
        print(f"[round {r:02d}] (FedMD mean) hamming={mean['hamming_acc']:.3f} "
              f"macro_f1={mean['macro_f1']:.3f} auroc={mean['auroc']:.3f} "
              f"| {secs:.1f}s | comm={cum_comm_mb:.2f}MB cumulative")
        mean["_comm_mb"] = cum_comm_mb
        _log_round(r, mean, secs)
    print(f"\ntotal {time.time()-t0:.1f}s")
    _csv_f.close()
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
