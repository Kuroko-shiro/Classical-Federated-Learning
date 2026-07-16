"""Phase-0 runner for scenario 4 (width and modality heterogeneity)."""

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
from qflbench.diagnostics.drift import update_drift_report
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
from qflbench.experiments.iu_runtime import (
    RunArtifacts,
    canonical_client_evaluation,
    mean_metrics,
    rare_labels_from_manifest,
)
from qflbench.models.base import hetero_aggregate, slice_to
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


def parse_ratio(value, clients):
    multimodal, unimodal = (int(part) for part in value.split(":"))
    if multimodal + unimodal != clients:
        raise ValueError(f"--mm-ratio {value} must sum to --clients {clients}")
    return multimodal, unimodal


def default_embed_dims(clients):
    palette = [128, 256, 192, 320, 224, 288, 160, 352]
    return [palette[index % len(palette)] for index in range(clients)]


def client_models(clients):
    return {
        f"client_{ctx.client_id}": backend.get_parameters(only_shared=True)
        for ctx, backend, _ in clients
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", default="heterofl", choices=["heterofl", "fedmd", "fedmd_loot"])
    ap.add_argument("--mm-ratio", default="1:3")
    ap.add_argument("--split", default="splits/iu_split.json")
    ap.add_argument("--clients", type=int, default=4)
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
    ap.add_argument("--use-min", action="store_true")
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument(
        "--diagnostics", action="store_true",
        help="save client-update drift and MIN pathway diagnostics",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument(
        "--embed-dims", type=int, nargs="+", default=None,
        help="per-client widths; default for four clients is 128 256 192 320",
    )
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--results-root", default="results/iu")
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    print(f"device={pick_device()} scenario=4 method={args.method} seed={args.seed}")

    manifest = build_manifest(
        args.reports, args.projections, args.images,
        require_findings=True, require_frontal=False,
    )
    protocol = load_iu_split(
        args.split, manifest_size=len(manifest), alpha=args.alpha,
        clients=args.clients, train_subset=args.train_subset,
        test_subset=args.test_subset, val_fraction=args.val_fraction,
        val_seed=args.val_seed,
        labels=np.stack([item["label"] for item in manifest]),
    )
    part = protocol.train_by_client
    print("protocol:", protocol.provenance())

    embed_dims = args.embed_dims or default_embed_dims(args.clients)
    if len(embed_dims) != args.clients:
        raise ValueError("--embed-dims must contain exactly --clients values")
    multimodal, unimodal = parse_ratio(args.mm_ratio, args.clients)
    client_mods = [["image", "text"]] * multimodal + [["image"]] * unimodal
    if args.use_min and args.method != "heterofl":
        raise ValueError("--use-min is currently a HeteroFL diagnostic")
    if args.use_min and max(embed_dims[:multimodal]) != max(embed_dims):
        raise ValueError(
            "MIN slicing requires a maximum-width multimodal client; use "
            "--embed-dims 320 256 192 128 for M2"
        )
    print("embed dims:", embed_dims)
    print("client modalities:", {cid: mods for cid, mods in enumerate(client_mods)})

    img_cache = torch.load(args.img_cache) if args.img_cache else None
    tok = AutoTokenizer.from_pretrained(args.text_model)
    val_loaders = {
        cid: make_loader(
            manifest, protocol.validation, tok, args.img_size, args.batch, False,
            client_mods[cid], img_cache=img_cache, num_workers=args.num_workers,
        )
        for cid in range(args.clients)
    }
    test_loaders = {
        cid: make_loader(
            manifest, protocol.test, tok, args.img_size, args.batch, False,
            client_mods[cid], img_cache=img_cache, num_workers=args.num_workers,
        )
        for cid in range(args.clients)
    }
    audit_loader = make_loader(
        manifest, protocol.validation, tok, args.img_size, args.batch, False,
        ["image", "text"], img_cache=img_cache, num_workers=args.num_workers,
    )
    train_pool = sorted({
        index
        for values in list(protocol.train_by_client.values()) + list(protocol.val_by_client.values())
        for index in values
    })
    rare_labels = rare_labels_from_manifest(manifest, train_pool)
    public_loaders = None
    if args.method != "heterofl":
        public_loaders = {
            cid: make_loader(
                manifest, protocol.public, tok, args.img_size, args.batch, False,
                client_mods[cid], img_cache=img_cache, num_workers=args.num_workers,
            )
            for cid in range(args.clients)
        }

    clients = []
    for cid in range(args.clients):
        ctx = _Ctx(cid, 14, client_mods[cid], len(part[cid]))
        backend = TorchMultimodalBackend(
            ctx, dataset=None, embed_dim=embed_dims[cid], share_encoders=True,
            all_modalities=True, use_min=args.use_min, text_model=args.text_model,
            pretrained=True, seed=args.seed,
        )
        loader = make_loader(
            manifest, part[cid], tok, args.img_size, args.batch, True,
            client_mods[cid], img_cache=img_cache,
            num_workers=args.num_workers, seed=args.seed + cid,
        )
        clients.append((ctx, backend, loader))

    global_backend = None
    if args.method == "heterofl":
        global_backend = TorchMultimodalBackend(
            _Ctx(-1, 14, ["image", "text"], 0), dataset=None,
            embed_dim=max(embed_dims), share_encoders=True, all_modalities=True,
            use_min=args.use_min, text_model=args.text_model, pretrained=True,
            seed=args.seed + 999,
        )

    artifacts = RunArtifacts(
        root=args.results_root,
        run_name=f"s4_{args.method}_{args.mm_ratio.replace(':', '-')}_a{args.alpha}_seed{args.seed}",
        config={**vars(args), "scenario": 4, "embed_dims": embed_dims,
                "client_modalities": client_mods},
        protocol=protocol.provenance(),
        repo_dir=os.path.join(os.path.dirname(__file__), ".."),
    )
    checkpoint = BestCheckpoint(str(artifacts.checkpoint_path), metric="auroc")
    ledger = CommunicationLedger()
    checkpoint_meta = {
        "runner": "iu_federated_s4.py", "scenario": 4, "method": args.method,
        "embed_dims": embed_dims, "client_modalities": client_mods,
        "args": vars(args), "protocol": protocol.provenance(),
    }

    try:
        if args.use_min:
            min_sources = [row for row in clients if "text" in row[0].modalities]
            ledger.start_round(-1)
            min_states, min_weights, pretrain_rows = [], [], []
            for ctx, backend, loader in min_sources:
                diagnostic = backend.pretrain_min(
                    loader, epochs=args.min_epochs, lr=1e-3,
                )
                state = backend.get_min_parameters()
                ledger.record(
                    ctx.client_id, "upload", state, payload_type="min_parameter",
                    metadata={"phase": "pretraining"},
                )
                min_states.append(state)
                min_weights.append(ctx.num_train)
                pretrain_rows.append({"client_id": ctx.client_id, **diagnostic})
            global_min = hetero_aggregate(min_states, min_weights)
            for ctx, backend, _ in clients:
                local_min = slice_to(global_min, backend.get_min_parameters())
                ledger.record(
                    ctx.client_id, "download", local_min,
                    payload_type="nested_min_parameter",
                    metadata={"phase": "pretraining"},
                )
                backend.set_min_parameters(local_min)
            pretrain_communication = ledger.finish_round()
            pretrain_communication["phase"] = "min_pretraining"
            artifacts.log_communication(pretrain_communication)
            artifacts.write_json("diagnostics/min_pretraining.json", pretrain_rows)
            artifacts.write_json(
                "diagnostics/min_reconstruction.json",
                clients[0][1].min_reconstruction_diagnostics(audit_loader),
            )

        if args.method == "heterofl":
            gp = global_backend.get_parameters(only_shared=True)
            for round_index in range(args.rounds):
                lap = time.time()
                ledger.start_round(round_index)
                updates, references, weights = [], [], []
                min_round = []
                for ctx, backend, loader in clients:
                    reference = backend.get_parameters(only_shared=True)
                    submodel = slice_to(gp, reference)
                    ledger.record(
                        ctx.client_id, "download", submodel,
                        payload_type="nested_parameter",
                    )
                    backend.set_parameters(submodel, only_shared=True)
                    train_metrics = backend.local_train(
                        loader, epochs=args.local_epochs, lr=args.lr,
                    )
                    update = backend.get_parameters(only_shared=True)
                    ledger.record(
                        ctx.client_id, "upload", update,
                        payload_type="nested_parameter",
                    )
                    updates.append(update)
                    references.append(submodel)
                    weights.append(ctx.num_train)
                    if args.use_min:
                        min_round.append({"client_id": ctx.client_id, **{
                            key: value for key, value in train_metrics.items()
                            if key.startswith("min_")
                        }})
                gp = hetero_aggregate(updates, weights)
                if args.diagnostics:
                    drift = update_drift_report(
                        updates, references, weights,
                        client_ids=[ctx.client_id for ctx, _, _ in clients],
                    )
                    artifacts.append_jsonl(
                        "diagnostics/update_drift.jsonl",
                        {"round": round_index, **drift},
                    )
                if args.use_min:
                    artifacts.append_jsonl(
                        "diagnostics/min_training.jsonl",
                        {"round": round_index, "clients": min_round},
                    )
                validation_rows = []
                for ctx, backend, _ in clients:
                    backend.set_parameters(
                        slice_to(gp, backend.get_parameters(only_shared=True)),
                        only_shared=True,
                    )
                    validation_rows.append(
                        backend.evaluate(val_loaders[ctx.client_id], rare_labels=rare_labels)
                    )
                validation = mean_metrics(validation_rows)
                communication = ledger.finish_round()
                artifacts.log_validation(
                    round_index, validation, time.time() - lap, communication,
                )
                checkpoint_models = {"global": gp}
                if args.use_min:
                    checkpoint_models.update({
                        f"min_client_{ctx.client_id}": backend.get_min_parameters()
                        for ctx, backend, _ in clients
                    })
                checkpoint.update(
                    round_index, validation, checkpoint_models, metadata=checkpoint_meta,
                )
                print(f"[R{round_index:02d}] val_auroc={validation['auroc']:.4f} "
                      f"comm={communication['cumulative_total_bytes']/1e6:.1f}MB")
        else:
            for round_index in range(args.rounds):
                lap = time.time()
                ledger.start_round(round_index)
                logits = []
                for ctx, backend, loader in clients:
                    backend.local_train(loader, epochs=args.local_epochs, lr=args.lr)
                    client_logits = backend.predict_logits(public_loaders[ctx.client_id])
                    ledger.record(ctx.client_id, "upload", client_logits, payload_type="logit")
                    logits.append(client_logits)
                stacked = np.stack(logits, axis=0)
                for position, (ctx, backend, _) in enumerate(clients):
                    teacher = (
                        stacked.mean(axis=0) if args.method == "fedmd"
                        else np.delete(stacked, position, axis=0).mean(axis=0)
                    )
                    ledger.record(ctx.client_id, "download", teacher, payload_type="logit")
                    backend.distill(
                        public_loaders[ctx.client_id], teacher,
                        epochs=args.distill_epochs,
                        lr=args.lr, temperature=args.distill_temp,
                    )
                validation = mean_metrics([
                    backend.evaluate(val_loaders[ctx.client_id], rare_labels=rare_labels)
                    for ctx, backend, _ in clients
                ])
                communication = ledger.finish_round()
                artifacts.log_validation(
                    round_index, validation, time.time() - lap, communication,
                )
                checkpoint.update(
                    round_index, validation, client_models(clients),
                    metadata=checkpoint_meta,
                )
                print(f"[R{round_index:02d}] val_auroc={validation['auroc']:.4f} "
                      f"comm={communication['cumulative_total_bytes']/1e6:.4f}MB")

        selected, models, _ = load_checkpoint(str(artifacts.checkpoint_path))
        if args.method == "heterofl":
            gp = models["global"]
            evaluation_rows = []
            for ctx, backend, _ in clients:
                backend.set_parameters(
                    slice_to(gp, backend.get_parameters(only_shared=True)),
                    only_shared=True,
                )
                if args.use_min:
                    backend.set_min_parameters(models[f"min_client_{ctx.client_id}"])
                evaluation_rows.append((
                    ctx.client_id, backend, val_loaders[ctx.client_id],
                    test_loaders[ctx.client_id],
                ))
        else:
            evaluation_rows = []
            for ctx, backend, _ in clients:
                backend.set_parameters(models[f"client_{ctx.client_id}"], only_shared=True)
                evaluation_rows.append((
                    ctx.client_id, backend, val_loaders[ctx.client_id],
                    test_loaders[ctx.client_id],
                ))
        test_metrics, details = canonical_client_evaluation(
            evaluation_rows, rare_labels=rare_labels,
        )
        if args.use_min:
            details["modality_ablation"] = {
                mode: clients[0][1].evaluate(
                    audit_loader, rare_labels=rare_labels, modality_mode=mode,
                )
                for mode in ("image_only", "true_text", "min_text", "zero_text", "text_only")
            }
        artifacts.write_test(
            metrics=test_metrics, checkpoint_metadata=selected, details=details,
        )
        print(f"selected round={selected['best_round']} "
              f"val_auroc={selected['best_validation_metrics']['auroc']:.4f}")
        print(f"test_auroc={test_metrics['auroc']:.4f} (single full-test evaluation)")
        print(f"run artifacts: {artifacts.run_dir}")
        print(f"total seconds: {time.time() - started:.1f}")
    finally:
        artifacts.close()


if __name__ == "__main__":
    main()
