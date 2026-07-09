"""Compute FedAvg communication volume WITHOUT re-running training.

FedAvg sends a FIXED-size payload (the shared parameters) up and down every round,
so total bytes = param_bytes * 2 * num_clients * num_rounds. We just need the
shared-parameter byte count, which we get by instantiating the model once and
measuring it. This does NOT touch any running job and needs no GPU compute.

Run on the M5 (safe to run while training is in progress):
    python scripts/iu_comm_estimate.py --clients 4 --rounds 40
    python scripts/iu_comm_estimate.py --clients 4 --rounds 40 --share-head-only

--share-head-only models scenario ③④ where only fusion+head is shared (encoders
stay local), giving a much smaller per-round payload.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


class _Ctx:
    def __init__(self):
        self.client_id = 0
        self.num_classes = 14
        self.modalities = ["image", "text"]


def _bytes_of(params: dict) -> int:
    total = 0
    for v in params.values():
        arr = np.asarray(v)
        total += arr.size * arr.itemsize
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--share-head-only", action="store_true",
                    help="scenario 3/4: only fusion+head shared (encoders local)")
    args = ap.parse_args()

    from qflbench.models.iu_xray_torch_model import TorchMultimodalBackend

    ctx = _Ctx()
    backend = TorchMultimodalBackend(
        ctx, dataset=None, embed_dim=args.embed_dim,
        share_encoders=(not args.share_head_only),
        text_model=args.text_model, pretrained=False)  # weights irrelevant for size

    shared = backend.get_parameters(only_shared=True)
    n_params = sum(int(np.asarray(v).size) for v in shared.values())
    param_bytes = _bytes_of(shared)

    # FedAvg: each round, every client uploads AND downloads the shared params
    per_round = param_bytes * 2 * args.clients
    total = per_round * args.rounds

    def fmt(b):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"

    print("=== FedAvg communication estimate ===")
    print(f"  mode: {'fusion+head only' if args.share_head_only else 'full model (all shared)'}")
    print(f"  shared params: {n_params:,} ({fmt(param_bytes)} as float32)")
    print(f"  clients: {args.clients}, rounds: {args.rounds}")
    print(f"  per-round (up+down, all clients): {fmt(per_round)}")
    print(f"  TOTAL over {args.rounds} rounds: {fmt(total)}")
    print()
    print("  note: this is the classical baseline the QFL comparison needs.")
    print("  FedMD/LOOT differ (they send logits/embeddings, not params) and are")
    print("  measured live in the federated runner instead.")


if __name__ == "__main__":
    main()
