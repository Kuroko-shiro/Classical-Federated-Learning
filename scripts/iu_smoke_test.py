"""M5 smoke test: does the real-data multimodal pipeline run end-to-end?

NOT federated yet — just centralized training on a small subset, to confirm that
data loading + ResNet-50 + BERT + fusion + head + multi-label loss all work on
your machine before we layer FL on top.

Run on the M5 (NOT in the dev sandbox — needs torch/torchvision/transformers):
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/iu_smoke_test.py \
        --reports data/indiana_reports.csv \
        --projections data/indiana_projections.csv \
        --images data/images/images_normalized \
        --subset 200 --epochs 1 --batch 8

Expectation: loss prints and decreases a bit; eval metrics print without error.
If this passes, the encoders/fusion/head and the data path are correct.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qflbench.data.iu_xray_prep import (build_manifest, split_train_test,
                                        manifest_stats)
from qflbench.data.iu_xray_torch import IUXrayDataset
from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend, pick_device


class _Ctx:
    """Minimal stand-in for ClientContext (centralized => one 'client')."""
    def __init__(self, num_classes):
        self.client_id = 0
        self.num_classes = num_classes
        self.modalities = ["image", "text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True, help="image_root (images_normalized)")
    ap.add_argument("--subset", type=int, default=200, help="use first N samples")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    print(f"device = {pick_device()}")

    manifest = build_manifest(args.reports, args.projections, args.images,
                              require_findings=True, require_frontal=False)
    stats = manifest_stats(manifest)
    print(f"manifest: {stats['n_samples']} samples, "
          f"both views={stats['has_both']}, avg_labels={stats['avg_labels']:.2f}")

    split = split_train_test(manifest, test_frac=0.2, seed=0)
    train_idx = split["train"][:args.subset]
    test_idx = split["test"][:max(args.subset // 4, 16)]

    tok = AutoTokenizer.from_pretrained(args.text_model)
    train_ds = IUXrayDataset(manifest, train_idx, tok, img_size=args.img_size,
                             train=True)
    test_ds = IUXrayDataset(manifest, test_idx, tok, img_size=args.img_size,
                            train=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    ctx = _Ctx(num_classes=14)
    backend = TorchMultimodalBackend(ctx, dataset=None, embed_dim=256,
                                     share_encoders=True,
                                     text_model=args.text_model, pretrained=True)

    print("training...")
    m = backend.local_train(train_ld, epochs=args.epochs, lr=1e-4)
    print(f"final train loss: {m['loss']:.4f}")

    print("evaluating...")
    metrics = backend.evaluate(test_ld)
    print("test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nOK: real-data multimodal pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
