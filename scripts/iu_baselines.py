"""Validation-selected centralized upper and local lower baselines."""

from __future__ import annotations

import argparse
import math
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
    load_checkpoint,
    load_iu_split,
)
from qflbench.experiments.iu_runtime import RunArtifacts, mean_metrics
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device


ZERO_COMMUNICATION = {
    "upload_bytes": 0, "download_bytes": 0, "total_bytes": 0,
    "cumulative_upload_bytes": 0, "cumulative_download_bytes": 0,
    "cumulative_total_bytes": 0, "clients": {},
}


class _Ctx:
    def __init__(self, cid, num_classes, modalities, n_train):
        self.client_id = cid
        self.num_classes = num_classes
        self.modalities = modalities
        self.n_train = n_train


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


def learning_rate(base, epoch, epochs, decay):
    if decay == "cosine":
        return base * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs)))
    return base


def client_modalities(clients, ratio):
    if ratio is None:
        return [["image", "text"]] * clients
    multimodal, unimodal = (int(part) for part in ratio.split(":"))
    if multimodal + unimodal != clients:
        raise ValueError("--mm-ratio must sum to --clients")
    return [["image", "text"]] * multimodal + [["image"]] * unimodal


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--mode", required=True, choices=["centralized", "local"])
    ap.add_argument("--split", default="splits/iu_split.json")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-subset", type=int, default=DEFAULT_TRAIN_SUBSET)
    ap.add_argument("--test-subset", type=int, default=DEFAULT_TEST_SUBSET)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--lr", type=float, default=None,
        help="default: 3e-5 centralized, 1e-4 local",
    )
    ap.add_argument("--lr-decay", choices=["none", "cosine"], default="none")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--vary-embed", action="store_true")
    ap.add_argument("--mm-ratio", default=None)
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--results-root", default="results/iu")
    return ap.parse_args()


def main():
    args = parse_args()
    effective_lr = args.lr if args.lr is not None else (3e-5 if args.mode == "centralized" else 1e-4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    print(f"device={pick_device()} mode={args.mode} seed={args.seed} lr={effective_lr}")

    manifest = build_manifest(
        args.reports, args.projections, args.images,
        require_findings=True, require_frontal=False,
    )
    protocol = load_iu_split(
        args.split, manifest_size=len(manifest), alpha=args.alpha,
        clients=args.clients, train_subset=args.train_subset,
        test_subset=args.test_subset, val_fraction=args.val_fraction,
        val_seed=args.val_seed,
    )
    print("protocol:", protocol.provenance())
    img_cache = torch.load(args.img_cache) if args.img_cache else None
    tok = AutoTokenizer.from_pretrained(args.text_model)
    test_loader = make_loader(
        manifest, protocol.test, tok, args.img_size, args.batch, False,
        ["image", "text"], img_cache=img_cache, num_workers=args.num_workers,
    )

    run_config = {**vars(args), "effective_lr": effective_lr}
    artifacts = RunArtifacts(
        root=args.results_root,
        run_name=f"baseline_{args.mode}_a{args.alpha}_seed{args.seed}",
        config=run_config, protocol=protocol.provenance(),
        repo_dir=os.path.join(os.path.dirname(__file__), ".."),
        extra_validation_fields=("client_id", "epoch"),
    )

    try:
        if args.mode == "centralized":
            train_indices = sorted({
                index for values in protocol.train_by_client.values() for index in values
            })
            ctx = _Ctx(0, 14, ["image", "text"], len(train_indices))
            backend = TorchMultimodalBackend(
                ctx, dataset=None, embed_dim=args.embed_dim, share_encoders=True,
                text_model=args.text_model, pretrained=True, seed=args.seed,
            )
            train_loader = make_loader(
                manifest, train_indices, tok, args.img_size, args.batch, True,
                ["image", "text"], img_cache=img_cache,
                num_workers=args.num_workers, seed=args.seed,
            )
            val_loader = make_loader(
                manifest, protocol.validation, tok, args.img_size, args.batch, False,
                ["image", "text"], img_cache=img_cache,
                num_workers=args.num_workers,
            )
            checkpoint = BestCheckpoint(str(artifacts.checkpoint_path), metric="auroc")
            metadata = {
                "runner": "iu_baselines.py", "mode": "centralized",
                "args": run_config, "protocol": protocol.provenance(),
            }
            for epoch in range(args.epochs):
                lap = time.time()
                lr = learning_rate(effective_lr, epoch, args.epochs, args.lr_decay)
                backend.local_train(train_loader, epochs=1, lr=lr)
                if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                    validation = backend.evaluate(val_loader)
                    artifacts.log_validation(
                        epoch, validation, time.time() - lap, ZERO_COMMUNICATION,
                        extra={"client_id": "all", "epoch": epoch},
                    )
                    checkpoint.update(
                        epoch, validation,
                        {"global": backend.get_parameters(only_shared=True)},
                        metadata=metadata,
                    )
                    print(f"[E{epoch:02d}] val_auroc={validation['auroc']:.4f} lr={lr:.2e}")
            selected, models, _ = load_checkpoint(str(artifacts.checkpoint_path))
            backend.set_parameters(models["global"], only_shared=True)
            test_metrics = backend.evaluate(test_loader)
            artifacts.write_test(metrics=test_metrics, checkpoint_metadata=selected)
            print(f"centralized C={test_metrics['auroc']:.4f} selected_epoch={selected['best_round']}")

        else:
            embed_dims = (
                [128, 256, 192, 320][:args.clients]
                if args.vary_embed else [args.embed_dim] * args.clients
            )
            if len(embed_dims) != args.clients:
                raise ValueError("the current heterogeneous-width recipe supports up to four clients")
            modalities = client_modalities(args.clients, args.mm_ratio)
            selected_clients = []
            test_rows = []
            for cid in range(args.clients):
                ctx = _Ctx(
                    cid, 14, modalities[cid], len(protocol.train_by_client[cid])
                )
                backend = TorchMultimodalBackend(
                    ctx, dataset=None, embed_dim=embed_dims[cid], share_encoders=True,
                    text_model=args.text_model, pretrained=True, seed=args.seed,
                )
                train_loader = make_loader(
                    manifest, protocol.train_by_client[cid], tok,
                    args.img_size, args.batch, True, modalities[cid],
                    img_cache=img_cache, num_workers=args.num_workers,
                    seed=args.seed + cid,
                )
                val_loader = make_loader(
                    manifest, protocol.val_by_client[cid], tok,
                    args.img_size, args.batch, False, ["image", "text"],
                    img_cache=img_cache, num_workers=args.num_workers,
                )
                client_checkpoint = BestCheckpoint(
                    str(artifacts.run_dir / f"best_validation_client_{cid}.npz"),
                    metric="auroc",
                )
                metadata = {
                    "runner": "iu_baselines.py", "mode": "local",
                    "client_id": cid, "embed_dim": embed_dims[cid],
                    "modalities": modalities[cid], "args": run_config,
                    "protocol": protocol.provenance(),
                }
                for epoch in range(args.epochs):
                    lap = time.time()
                    lr = learning_rate(effective_lr, epoch, args.epochs, args.lr_decay)
                    backend.local_train(train_loader, epochs=1, lr=lr)
                    if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                        validation = backend.evaluate(val_loader)
                        artifacts.log_validation(
                            cid * args.epochs + epoch, validation,
                            time.time() - lap, ZERO_COMMUNICATION,
                            extra={"client_id": cid, "epoch": epoch},
                        )
                        client_checkpoint.update(
                            epoch, validation,
                            {"client": backend.get_parameters(only_shared=True)},
                            metadata=metadata,
                        )
                selected, models, _ = load_checkpoint(str(client_checkpoint.path))
                backend.set_parameters(models["client"], only_shared=True)
                client_test = backend.evaluate(test_loader)
                selected_clients.append(selected)
                test_rows.append(client_test)
                print(f"client={cid} selected_epoch={selected['best_round']} "
                      f"test_auroc={client_test['auroc']:.4f}")
            test_metrics = mean_metrics(test_rows)
            artifacts.write_test(
                metrics=test_metrics,
                checkpoint_metadata={"per_client": selected_clients},
            )
            print(f"local lower bound={test_metrics['auroc']:.4f} (mean of validation-selected clients)")

        print(f"run artifacts: {artifacts.run_dir}")
        print(f"total seconds: {time.time() - started:.1f}")
    finally:
        artifacts.close()


if __name__ == "__main__":
    main()
