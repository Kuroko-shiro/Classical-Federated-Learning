"""Evaluate a Phase-0 NumPy checkpoint on the frozen full IU X-ray test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qflbench.data.iu_xray_prep import build_manifest
from qflbench.data.iu_xray_torch import IUXrayDataset
from qflbench.experiments.iu_protocol import DEFAULT_TEST_SUBSET, load_checkpoint, load_iu_split
from qflbench.experiments.iu_runtime import mean_metrics
from qflbench.models.base import slice_to
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend


class _Ctx:
    def __init__(self, cid, modalities):
        self.client_id = cid
        self.num_classes = 14
        self.modalities = modalities


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default=None, help="override checkpoint split path")
    ap.add_argument("--test-subset", type=int, default=DEFAULT_TEST_SUBSET)
    ap.add_argument("--img-cache", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def _backend(cid, dim, modalities, config, *, proto_dim=0):
    return TorchMultimodalBackend(
        _Ctx(cid, modalities), dataset=None, embed_dim=dim,
        share_encoders=True, all_modalities=True,
        text_model=config.get("text_model", "bert-base-uncased"),
        pretrained=True, proto_dim=proto_dim,
        seed=int(config.get("seed", 0)),
    )


def main():
    cli = parse_args()
    metadata, models, _ = load_checkpoint(cli.checkpoint)
    config = metadata.get("args", {})
    manifest = build_manifest(
        cli.reports, cli.projections, cli.images,
        require_findings=True, require_frontal=False,
    )
    clients = int(config.get("clients", 4))
    split_path = cli.split or config.get("split", "splits/iu_split.json")
    protocol = load_iu_split(
        split_path, manifest_size=len(manifest),
        alpha=float(config.get("alpha", 100.0)), clients=clients,
        train_subset=int(config.get("train_subset", 2510)),
        test_subset=cli.test_subset,
        val_fraction=float(config.get("val_fraction", 0.1)),
        val_seed=int(config.get("val_seed", 20260715)),
        all_data=bool(config.get("all_data", False)),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.get("text_model", "bert-base-uncased"))
    cache = torch.load(cli.img_cache) if cli.img_cache else None
    dataset = IUXrayDataset(
        manifest, protocol.test, tokenizer,
        img_size=int(config.get("img_size", 224)), train=False,
        modalities=["image", "text"], img_cache=cache,
    )
    loader = DataLoader(
        dataset, batch_size=int(config.get("batch", 8)), shuffle=False,
        num_workers=cli.num_workers,
    )

    method = metadata.get("method", config.get("method", "centralized"))
    embed_dims = metadata.get("embed_dims") or [int(config.get("embed_dim", 256))] * clients
    modalities = metadata.get("client_modalities") or [["image", "text"]] * clients
    if "global" in models and method != "heterofl":
        backend = _backend(-1, int(config.get("embed_dim", embed_dims[0])), ["image", "text"], config)
        backend.set_parameters(models["global"], only_shared=True)
        metrics = backend.evaluate(loader)
    elif "global" in models:  # nested-width HeteroFL
        global_params = models["global"]
        rows = []
        for cid, dim in enumerate(embed_dims):
            backend = _backend(cid, int(dim), modalities[cid], config)
            backend.set_parameters(
                slice_to(global_params, backend.get_parameters(only_shared=True)),
                only_shared=True,
            )
            rows.append(backend.evaluate(loader))
        metrics = mean_metrics(rows)
    elif "client" in models:  # one local-baseline checkpoint
        dim = int(metadata.get("embed_dim", config.get("embed_dim", 256)))
        backend = _backend(int(metadata.get("client_id", 0)), dim, metadata.get("modalities", ["image", "text"]), config)
        backend.set_parameters(models["client"], only_shared=True)
        metrics = backend.evaluate(loader)
    else:
        rows = []
        proto_dim = int(config.get("proto_dim", 0)) if method == "fedproto" else 0
        for cid, dim in enumerate(embed_dims):
            backend = _backend(cid, int(dim), modalities[cid], config, proto_dim=proto_dim)
            backend.set_parameters(models[f"client_{cid}"], only_shared=True)
            rows.append(backend.evaluate(loader))
        metrics = mean_metrics(rows)

    digest = hashlib.sha256()
    with open(cli.checkpoint, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_hash = digest.hexdigest()
    result = {
        "metrics": metrics,
        "test_size": len(protocol.test),
        "test_policy": "frozen full test, one evaluation",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata": metadata,
    }
    output = Path(cli.output) if cli.output else Path(cli.checkpoint).with_suffix(".reeval_test.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "test_size": len(protocol.test), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
