"""Pre-resize all IU X-ray images once and cache as tensors (speed, NOT quality).

The federated loop currently opens each PNG from disk and resizes it EVERY round.
Decoding + resizing thousands of images per round is pure overhead that does not
change the model at all. This script does it ONCE: load -> grayscale -> resize ->
save as a single .pt tensor cache keyed by filename. The training DataLoader can
then read pre-resized tensors instead of re-decoding PNGs.

This is a lossless speedup: identical pixels reach the model, just precomputed.

Run once on the M5 (re-run if you change --img-size):
    python scripts/iu_cache_images.py \
        --projections data/indiana_projections.csv \
        --images data/images/images_normalized \
        --img-size 224 \
        --out data/img_cache_224.pt

Output: a dict {filename: uint8 tensor [H,W]} saved with torch.save. Stored as
uint8 (0-255) to keep the cache small; normalization happens at load time.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projections", required=True)
    ap.add_argument("--images", required=True, help="image_root")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True, help="output .pt cache path")
    args = ap.parse_args()

    proj = pd.read_csv(args.projections)
    filenames = sorted(set(str(f) for f in proj["filename"]))
    print(f"{len(filenames)} unique images to cache at {args.img_size}px")

    resize = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.img_size, args.img_size)),
    ])

    cache = {}
    ok, miss = 0, 0
    for i, fn in enumerate(filenames):
        path = os.path.join(args.images, fn)
        try:
            img = Image.open(path).convert("L")
            img = resize(img)
            # store as uint8 [H,W]; bytearray makes a writable buffer (no warning)
            t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8
                                 ).reshape(args.img_size, args.img_size).clone()
            cache[fn] = t
            ok += 1
        except Exception as e:
            miss += 1
            if miss <= 5:
                print(f"  miss: {fn} ({e})")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(filenames)}")

    torch.save(cache, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"cached {ok} images ({miss} missing) -> {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
