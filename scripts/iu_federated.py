"""Phase-0 IU X-ray runner for scenarios 1 and 2.

The validation trajectory is written every round.  The full frozen test set is
evaluated exactly once, after restoring the checkpoint selected by validation
macro-AUROC.  This prevents the former test-peak leakage.
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

from qflbench.data.iu_xray_prep import build_manifest
from qflbench.data.iu_xray_torch import IUXrayDataset
from qflbench.experiments.iu_protocol import (
    DEFAULT_TEST_SUBSET,
    DEFAULT_TRAIN_SUBSET,
    DEFAULT_VAL_FRACTION,
    DEFAULT_VAL_SEED,
    BestCheckpoint,
    CommunicationLedger,
    load_checkpoint,
    load_iu_split,
)
from qflbench.experiments.iu_runtime import RunArtifacts, mean_metrics
from qflbench.models.base import hetero_aggregate, slice_to, weighted_average
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device


class _Ctx:
    def __init__(self, cid, num_classes, modalities, n_train):
        self.client_id = cid
        self.num_classes = num_classes
        self.modalities = modalities
        self.num_train = n_train


def make_loader(
    manifest, idx, tok, img_size, batch, train, modalities,
    *, img_cache=None, num_workers=0, seed=0,
):
    ds = IUXrayDataset(
        manifest, idx, tok, img_size=img_size, train=train,
        modalities=modalities, img_cache=img_cache,
    )
    generator = torch.Generator().manual_seed(int(seed)) if train else None
    return DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=num_workers,
        generator=generator,
    )


def _client_models(clients):
    return {
        f"client_{ctx.client_id}": backend.get_parameters(only_shared=True)
        for ctx, backend, _ in clients
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    ap.add_argument("--split", default="splits/iu_split.json")
    ap.add_argument(
        "--method", default="fedavg",
        choices=["fedavg", "fedprox", "fedmd", "heterofl", "fedproto"],
    )
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument(
        "--all-data", action="store_true",
        help="one-client centralized ceiling using the union of all frozen partitions",
    )
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--train-subset", type=int, default=DEFAULT_TRAIN_SUBSET)
    ap.add_argument("--test-subset", type=int, default=DEFAULT_TEST_SUBSET)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--distill-epochs", type=int, default=1)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--proto-dim", type=int, default=128)
    ap.add_argument("--proto-mu", type=float, default=0.1)
    ap.add_argument("--proto-warmup", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--freeze-text", action="store_true")
    ap.add_argument("--freeze-image", action="store_true")
    ap.add_argument("--results-root", default="results/iu")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.scenario == 1 and args.method in {"heterofl", "fedproto"}:
        raise ValueError(f"{args.method} is a scenario-2 method")
    if args.scenario == 2 and args.method in {"fedavg", "fedprox"}:
        raise ValueError("scenario 2 requires heterofl, fedmd, or fedproto")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"device={pick_device()} scenario={args.scenario} method={args.method} seed={args.seed}")
    started = time.time()

    manifest = build_manifest(
        args.reports, args.projections, args.images,
        require_findings=True, require_frontal=False,
    )
    protocol = load_iu_split(
        args.split, manifest_size=len(manifest), alpha=args.alpha,
        clients=args.clients, train_subset=args.train_subset,
        test_subset=args.test_subset, val_fraction=args.val_fraction,
        val_seed=args.val_seed, all_data=args.all_data,
    )
    part = protocol.train_by_client
    public_idx = protocol.public
    print("protocol:", protocol.provenance())

    img_cache = None
    if args.img_cache:
        print(f"loading image cache: {args.img_cache}")
        img_cache = torch.load(args.img_cache)
        print(f"cached images: {len(img_cache)}")

    tok = AutoTokenizer.from_pretrained(args.text_model)
    val_loader = make_loader(
        manifest, protocol.validation, tok, args.img_size, args.batch, False,
        ["image", "text"], img_cache=img_cache, num_workers=args.num_workers,
    )
    test_loader = make_loader(
        manifest, protocol.test, tok, args.img_size, args.batch, False,
        ["image", "text"], img_cache=img_cache, num_workers=args.num_workers,
    )

    embed_dims = (
        [args.embed_dim] * args.clients if args.scenario == 1
        else ([128, 256, 192, 320, 224, 256][:args.clients]
              + [256] * max(0, args.clients - 6))
    )
    clients = []
    for cid in range(args.clients):
        ctx = _Ctx(cid, 14, ["image", "text"], len(part[cid]))
        backend = TorchMultimodalBackend(
            ctx, dataset=None, embed_dim=embed_dims[cid], share_encoders=True,
            text_model=args.text_model, pretrained=True,
            freeze_text=args.freeze_text, freeze_image=args.freeze_image,
            proto_dim=(args.proto_dim if args.method == "fedproto" else 0),
            seed=args.seed,
        )
        loader = make_loader(
            manifest, part[cid], tok, args.img_size, args.batch, True,
            ["image", "text"], img_cache=img_cache,
            num_workers=args.num_workers, seed=args.seed + cid,
        )
        clients.append((ctx, backend, loader))

    global_backend = None
    if args.method in {"fedavg", "fedprox", "heterofl"}:
        gdim = max(embed_dims) if args.method == "heterofl" else args.embed_dim
        global_backend = TorchMultimodalBackend(
            _Ctx(-1, 14, ["image", "text"], 0), dataset=None,
            embed_dim=gdim, share_encoders=True, text_model=args.text_model,
            pretrained=True, freeze_text=args.freeze_text,
            freeze_image=args.freeze_image, seed=args.seed + 999,
        )

    artifacts = RunArtifacts(
        root=args.results_root,
        run_name=f"s{args.scenario}_{args.method}_a{args.alpha}_seed{args.seed}",
        config=vars(args), protocol=protocol.provenance(),
        repo_dir=os.path.join(os.path.dirname(__file__), ".."),
    )
    checkpoint = BestCheckpoint(str(artifacts.checkpoint_path), metric="auroc")
    ledger = CommunicationLedger()
    checkpoint_meta = {
        "runner": "iu_federated.py", "scenario": args.scenario,
        "method": args.method, "embed_dims": embed_dims,
        "args": vars(args), "protocol": protocol.provenance(),
    }

    try:
        if args.method in {"fedavg", "fedprox"}:
            gp = global_backend.get_parameters(only_shared=True)
            for round_index in range(args.rounds):
                lap = time.time()
                ledger.start_round(round_index)
                updates, weights = [], []
                for ctx, backend, loader in clients:
                    ledger.record(ctx.client_id, "download", gp)
                    backend.set_parameters(gp, only_shared=True)
                    backend.local_train(
                        loader, epochs=args.local_epochs, lr=args.lr,
                        proximal_mu=(args.mu if args.method == "fedprox" else 0.0),
                        global_params=(gp if args.method == "fedprox" else None),
                    )
                    update = backend.get_parameters(only_shared=True)
                    ledger.record(ctx.client_id, "upload", update)
                    updates.append(update)
                    weights.append(ctx.num_train)
                gp = weighted_average(updates, weights)
                global_backend.set_parameters(gp, only_shared=True)
                val = global_backend.evaluate(val_loader)
                comm = ledger.finish_round()
                artifacts.log_validation(round_index, val, time.time() - lap, comm)
                checkpoint.update(
                    round_index, val, {"global": gp}, metadata=checkpoint_meta,
                )
                print(f"[R{round_index:02d}] val_auroc={val['auroc']:.4f} "
                      f"comm={comm['cumulative_total_bytes']/1e6:.1f}MB")

        elif args.method == "heterofl":
            gp = global_backend.get_parameters(only_shared=True)
            for round_index in range(args.rounds):
                lap = time.time()
                ledger.start_round(round_index)
                updates, weights = [], []
                for ctx, backend, loader in clients:
                    reference = backend.get_parameters(only_shared=True)
                    submodel = slice_to(gp, reference)
                    ledger.record(ctx.client_id, "download", submodel)
                    backend.set_parameters(submodel, only_shared=True)
                    backend.local_train(loader, epochs=args.local_epochs, lr=args.lr)
                    update = backend.get_parameters(only_shared=True)
                    ledger.record(ctx.client_id, "upload", update)
                    updates.append(update)
                    weights.append(ctx.num_train)
                gp = hetero_aggregate(updates, weights)
                val_rows = []
                for ctx, backend, _ in clients:
                    reference = backend.get_parameters(only_shared=True)
                    backend.set_parameters(slice_to(gp, reference), only_shared=True)
                    val_rows.append(backend.evaluate(val_loader))
                val = mean_metrics(val_rows)
                comm = ledger.finish_round()
                artifacts.log_validation(round_index, val, time.time() - lap, comm)
                checkpoint.update(
                    round_index, val, {"global": gp}, metadata=checkpoint_meta,
                )
                print(f"[R{round_index:02d}] val_auroc={val['auroc']:.4f} "
                      f"comm={comm['cumulative_total_bytes']/1e6:.1f}MB")

        elif args.method == "fedproto":
            global_protos = None
            for round_index in range(args.rounds):
                lap = time.time()
                ledger.start_round(round_index)
                mu = 0.0 if round_index < args.proto_warmup else args.proto_mu
                sums = np.zeros((14, args.proto_dim), dtype=np.float64)
                counts = np.zeros(14, dtype=np.float64)
                for ctx, backend, loader in clients:
                    backend.local_train_proto(
                        loader, epochs=args.local_epochs, lr=args.lr,
                        mu=mu, global_protos=global_protos,
                    )
                    client_sums, client_counts = backend.label_prototype_stats(loader)
                    ledger.record(ctx.client_id, "upload", (client_sums, client_counts))
                    sums += client_sums
                    counts += client_counts
                global_protos = (sums / np.maximum(counts, 1.0)[:, None]).astype(np.float32)
                for ctx, _, _ in clients:
                    ledger.record(ctx.client_id, "download", global_protos)
                val = mean_metrics([backend.evaluate(val_loader) for _, backend, _ in clients])
                comm = ledger.finish_round()
                artifacts.log_validation(round_index, val, time.time() - lap, comm)
                checkpoint.update(
                    round_index, val, _client_models(clients),
                    metadata=checkpoint_meta, arrays={"global_prototypes": global_protos},
                )
                print(f"[R{round_index:02d}] val_auroc={val['auroc']:.4f} "
                      f"comm={comm['cumulative_total_bytes']/1e6:.4f}MB")

        else:  # FedMD
            public_loader = make_loader(
                manifest, public_idx, tok, args.img_size, args.batch, False,
                ["image", "text"], img_cache=img_cache,
                num_workers=args.num_workers,
            )
            for round_index in range(args.rounds):
                lap = time.time()
                ledger.start_round(round_index)
                logits = []
                for ctx, backend, loader in clients:
                    backend.local_train(loader, epochs=args.local_epochs, lr=args.lr)
                    client_logits = backend.predict_logits(public_loader)
                    ledger.record(ctx.client_id, "upload", client_logits)
                    logits.append(client_logits)
                consensus = np.mean(np.stack(logits, axis=0), axis=0)
                for ctx, backend, _ in clients:
                    ledger.record(ctx.client_id, "download", consensus)
                    backend.distill(
                        public_loader, consensus, epochs=args.distill_epochs,
                        lr=(args.distill_lr or args.lr), temperature=args.distill_temp,
                    )
                val = mean_metrics([backend.evaluate(val_loader) for _, backend, _ in clients])
                comm = ledger.finish_round()
                artifacts.log_validation(round_index, val, time.time() - lap, comm)
                checkpoint.update(
                    round_index, val, _client_models(clients), metadata=checkpoint_meta,
                )
                print(f"[R{round_index:02d}] val_auroc={val['auroc']:.4f} "
                      f"comm={comm['cumulative_total_bytes']/1e6:.4f}MB")

        selected, models, _ = load_checkpoint(str(artifacts.checkpoint_path))
        if args.method in {"fedavg", "fedprox"}:
            global_backend.set_parameters(models["global"], only_shared=True)
            test_metrics = global_backend.evaluate(test_loader)
        elif args.method == "heterofl":
            gp = models["global"]
            test_rows = []
            for _, backend, _ in clients:
                backend.set_parameters(
                    slice_to(gp, backend.get_parameters(only_shared=True)),
                    only_shared=True,
                )
                test_rows.append(backend.evaluate(test_loader))
            test_metrics = mean_metrics(test_rows)
        else:
            test_rows = []
            for ctx, backend, _ in clients:
                backend.set_parameters(models[f"client_{ctx.client_id}"], only_shared=True)
                test_rows.append(backend.evaluate(test_loader))
            test_metrics = mean_metrics(test_rows)
        artifacts.write_test(metrics=test_metrics, checkpoint_metadata=selected)
        print(f"selected round={selected['best_round']} "
              f"val_auroc={selected['best_validation_metrics']['auroc']:.4f}")
        print(f"test_auroc={test_metrics['auroc']:.4f} (single full-test evaluation)")
        print(f"run artifacts: {artifacts.run_dir}")
        print(f"total seconds: {time.time() - started:.1f}")
    finally:
        artifacts.close()


if __name__ == "__main__":
    main()
