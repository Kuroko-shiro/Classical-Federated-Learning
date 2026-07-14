"""PyTorch Dataset for the IU X-ray image/findings manifest."""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms import functional as TF
except Exception as exc:  # pragma: no cover - exercised in the MPS environment
    raise ImportError(
        "IUXrayDataset requires the 'torch' optional dependencies: "
        "pip install -e '.[torch]'"
    ) from exc


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


class IUXrayDataset(Dataset):
    """One report/study per item with averaged image projections.

    Missing modalities still return shape-stable zero tensors plus
    ``has_image``/``has_text`` flags; ``MultimodalNet.encode`` uses those flags
    to exclude absent inputs from the fusion mean.
    """

    def __init__(
        self,
        manifest: Sequence[Mapping[str, object]],
        indices: Iterable[int],
        tokenizer,
        *,
        img_size: int = 224,
        max_length: int = 256,
        train: bool = False,
        modalities: Optional[Sequence[str]] = None,
        img_cache: Optional[Mapping[str, "torch.Tensor"]] = None,
    ):
        self.manifest = manifest
        self.indices = [int(i) for i in indices]
        self.tokenizer = tokenizer
        self.img_size = int(img_size)
        self.max_length = int(max_length)
        self.train = bool(train)
        self.modalities = set(modalities or ("image", "text"))
        unknown = self.modalities.difference({"image", "text"})
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        self.img_cache = img_cache

    def __len__(self) -> int:
        return len(self.indices)

    def _image_tensor(self, item: Mapping[str, object]) -> "torch.Tensor":
        tensors = []
        filenames = list(item.get("filenames", []))
        paths = list(item.get("image_paths", []))
        for offset, filename in enumerate(filenames):
            cached = self.img_cache.get(str(filename)) if self.img_cache is not None else None
            if cached is not None:
                tensor = torch.as_tensor(cached).to(torch.float32)
                if tensor.ndim == 3:
                    tensor = tensor.squeeze(0)
                tensor = tensor / 255.0
            else:
                path = paths[offset] if offset < len(paths) else str(filename)
                with Image.open(os.fspath(path)) as image:
                    image = image.convert("L")
                    image = TF.resize(image, [self.img_size, self.img_size], antialias=True)
                    tensor = TF.pil_to_tensor(image).squeeze(0).to(torch.float32) / 255.0
            if tensor.shape != (self.img_size, self.img_size):
                tensor = TF.resize(tensor.unsqueeze(0), [self.img_size, self.img_size], antialias=True).squeeze(0)
            tensors.append(tensor)
        if not tensors:
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        image = torch.stack(tensors).mean(dim=0).unsqueeze(0).repeat(3, 1, 1)
        return (image - _IMAGENET_MEAN) / _IMAGENET_STD

    def __getitem__(self, offset: int) -> dict:
        item = self.manifest[self.indices[offset]]
        has_image = "image" in self.modalities and bool(item.get("filenames"))
        findings = str(item.get("findings", ""))
        has_text = "text" in self.modalities and bool(findings.strip())

        image = self._image_tensor(item) if has_image else torch.zeros(
            (3, self.img_size, self.img_size), dtype=torch.float32
        )
        tokens = self.tokenizer(
            findings if has_text else "",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.as_tensor(np.asarray(item["label"]), dtype=torch.float32),
            "has_image": torch.tensor(has_image, dtype=torch.bool),
            "has_text": torch.tensor(has_text, dtype=torch.bool),
            "sample_index": torch.tensor(self.indices[offset], dtype=torch.long),
        }
