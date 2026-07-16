"""Phase-0 runner for scenario 3 (modality incongruity)."""

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
from qflbench.experiments.iu_runtime import (
    RunArtifacts,
    canonical_client_evaluation,
    mean_metrics,
    rare_labels_from_manifest,
)
from qflbench.models.base import weighted_average
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", default="fedavg", choices=["uniml", "fedavg", "min", "loot"])
    ap.add_argument("--mm-ratio", default="1:3")
    ap.add_argument("--split", default="splits/iu_split.json")
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--train-subset", type=int, default=DEFAULT_TRAIN_SUBSET)
    ap.add_argument("--test-subset", type=int, default=DEFAULT_TEST_SUBSET)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--results-root", default="results/iu")
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    print(f"device={pick_device()} scenario=3 method={args.method} seed={args.seed}")

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

    if args.method == "uniml":
        client_mods = [["image"]] * args.clients
    else:
        multimodal, unimodal = parse_ratio(args.mm_ratio, args.clients)
        client_mods = [["image", "text"]] * multimodal + [["image"]] * unimodal
    print("client modalities:", {cid: mods for cid, mods in enumerate(client_mods)})
    use_min = args.method == "min"

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
    # Public data replaces the former test-set LOOT probe.  Test features must
    # never influence training, even when labels are not read.
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
            ctx, dataset=None, embed_dim=args.embed_dim, share_encoders=True,
            all_modalities=True, use_min=use_min, text_model=args.text_model,
            pretrained=True, seed=args.seed,
        )
        loader = make_loader(
            manifest, part[cid], tok, args.img_size, args.batch, True,
            client_mods[cid], img_cache=img_cache,
            num_workers=args.num_workers, seed=args.seed + cid,
        )
        clients.append((ctx, backend, loader))

    global_backend = TorchMultimodalBackend(
        _Ctx(-1, 14, ["image", "text"], 0), dataset=None,
        embed_dim=args.embed_dim, share_encoders=True, all_modalities=True,
        use_min=use_min, text_model=args.text_model, pretrained=True,
        seed=args.seed + 999,
    )

    artifacts = RunArtifacts(
        root=args.results_root,
        run_name=f"s3_{args.method}_{args.mm_ratio.replace(':', '-')}_a{args.alpha}_seed{args.seed}",
        config={**vars(args), "scenario": 3, "client_modalities": client_mods},
        protocol=protocol.provenance(),
        repo_dir=os.path.join(os.path.dirname(__file__), ".."),
    )
    checkpoint = BestCheckpoint(str(artifacts.checkpoint_path), metric="auroc")
    ledger = CommunicationLedger()
    checkpoint_meta = {
        "runner": "iu_federated_s3.py", "scenario": 3, "method": args.method,
        "client_modalities": client_mods, "args": vars(args),
        "protocol": protocol.provenance(),
    }

    try:
        if use_min:
            multimodal_clients = [row for row in clients if "text" in row[0].modalities]
            if not multimodal_clients:
                raise RuntimeError("MIN requires at least one multimodal client")
            ledger.start_round(-1)
            min_states = []
            pretrain_diagnostics = []
            for ctx, backend, loader in multimodal_clients:
                diagnostic = backend.pretrain_min(loader, epochs=args.min_epochs, lr=1e-3)
                pretrain_diagnostics.append({"client_id": ctx.client_id, **diagnostic})
                state = {
                    key: value.detach().cpu().numpy()
                    for key, value in backend.net.min_net.state_dict().items()
                }
                ledger.record(ctx.client_id, "upload", state, payload_type="min_parameter")
                min_states.append(state)
            average_min = {
                key: np.mean([state[key] for state in min_states], axis=0)
                for key in min_states[0]
            }
            for ctx, backend, _ in clients:
                ledger.record(ctx.client_id, "download", average_min, payload_type="min_parameter")
                state = backend.net.min_net.state_dict()
                for key in state:
                    state[key] = torch.tensor(average_min[key], device=backend.device)
                backend.net.min_net.load_state_dict(state)
            global_state = global_backend.net.min_net.state_dict()
            for key in global_state:
                global_state[key] = torch.tensor(average_min[key], device=global_backend.device)
            global_backend.net.min_net.load_state_dict(global_state)
            pretrain_comm = ledger.finish_round()
            pretrain_comm["phase"] = "min_pretraining"
            artifacts.log_communication(pretrain_comm)
            artifacts.write_json(
                "diagnostics/min_pretraining.json", pretrain_diagnostics,
            )
            artifacts.write_json(
                "diagnostics/min_reconstruction.json",
                global_backend.min_reconstruction_diagnostics(audit_loader),
            )

        gp = global_backend.get_parameters(only_shared=True)
        for round_index in range(args.rounds):
            lap = time.time()
            ledger.start_round(round_index)
            updates, weights = [], []
            min_round = []
            for ctx, backend, loader in clients:
                ledger.record(ctx.client_id, "download", gp, payload_type="full_parameter")
                backend.set_parameters(gp, only_shared=True)
                train_metrics = backend.local_train(
                    loader, epochs=args.local_epochs, lr=args.lr,
                )
                if use_min:
                    min_round.append({"client_id": ctx.client_id, **{
                        key: value for key, value in train_metrics.items()
                        if key.startswith("min_")
                    }})
                update = backend.get_parameters(only_shared=True)
                ledger.record(ctx.client_id, "upload", update, payload_type="full_parameter")
                updates.append(update)
                weights.append(ctx.num_train)
            gp = weighted_average(updates, weights)
            global_backend.set_parameters(gp, only_shared=True)

            if args.method == "loot":
                embeddings = []
                for ctx, backend, _ in clients:
                    ledger.record(ctx.client_id, "download", gp, payload_type="full_parameter")
                    backend.set_parameters(gp, only_shared=True)
                    client_embeddings = backend.embed(public_loaders[ctx.client_id])
                    ledger.record(ctx.client_id, "upload", client_embeddings, payload_type="embedding")
                    embeddings.append(client_embeddings)
                stacked = np.stack(embeddings, axis=0)
                aligned_updates = []
                for position, (ctx, backend, _) in enumerate(clients):
                    teacher = np.delete(stacked, position, axis=0).mean(axis=0)
                    ledger.record(ctx.client_id, "download", teacher, payload_type="embedding")
                    backend.align_embeddings(
                        public_loaders[ctx.client_id], teacher, epochs=1, lr=args.lr
                    )
                    update = backend.get_parameters(only_shared=True)
                    ledger.record(ctx.client_id, "upload", update, payload_type="full_parameter")
                    aligned_updates.append(update)
                gp = weighted_average(aligned_updates, weights)
                global_backend.set_parameters(gp, only_shared=True)

            validation_rows = []
            for ctx, backend, _ in clients:
                backend.set_parameters(gp, only_shared=True)
                validation_rows.append(backend.evaluate(val_loaders[ctx.client_id]))
            validation = mean_metrics(validation_rows)
            communication = ledger.finish_round()
            artifacts.log_validation(
                round_index, validation, time.time() - lap, communication,
            )
            checkpoint_models = {"global": gp}
            if use_min:
                checkpoint_models.update({
                    f"min_client_{ctx.client_id}": backend.get_min_parameters()
                    for ctx, backend, _ in clients
                })
                artifacts.append_jsonl(
                    "diagnostics/min_training.jsonl",
                    {"round": round_index, "clients": min_round},
                )
            checkpoint.update(
                round_index, validation, checkpoint_models, metadata=checkpoint_meta,
            )
            print(f"[R{round_index:02d}] val_auroc={validation['auroc']:.4f} "
                  f"comm={communication['cumulative_total_bytes']/1e6:.1f}MB")

        selected, models, _ = load_checkpoint(str(artifacts.checkpoint_path))
        gp = models["global"]
        evaluation_rows = []
        for ctx, backend, _ in clients:
            backend.set_parameters(gp, only_shared=True)
            if use_min:
                backend.set_min_parameters(models[f"min_client_{ctx.client_id}"])
            evaluation_rows.append((
                ctx.client_id, backend, val_loaders[ctx.client_id],
                test_loaders[ctx.client_id],
            ))
        test_metrics, details = canonical_client_evaluation(
            evaluation_rows, rare_labels=rare_labels,
        )
        if use_min:
            audit_backend = clients[0][1]
            details["modality_ablation"] = {
                mode: audit_backend.evaluate(
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
