"""IU X-ray manifest construction and deterministic partitioning.

The Kaggle mirror used by this project exposes ``indiana_reports.csv`` and
``indiana_projections.csv``.  A manifest item represents one report/study and
contains its findings text, one or more image filenames and a 14-dimensional
weak-label vector.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .iu_xray_labels import CHEXPERT_LABELS, encode_problems


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    by_lower = {str(name).lower(): str(name) for name in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"required CSV column missing; expected one of {candidates}, "
        f"found {list(frame.columns)}"
    )


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _uid(value: object) -> str:
    """Normalise pandas' int/float inference differences across the two CSVs."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return _clean_text(value)


def build_manifest(
    reports_csv: str,
    projections_csv: str,
    image_root: str,
    *,
    require_findings: bool = True,
    require_frontal: bool = False,
) -> List[dict]:
    """Build a stable, UID-sorted study manifest from the IU X-ray CSVs.

    Images are kept at filename level and loaded lazily by ``IUXrayDataset``.
    When multiple projections exist for a study, the dataset averages the view
    tensors, matching the project protocol.
    """

    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)
    r_uid = _column(reports, "uid")
    p_uid = _column(projections, "uid")
    filename_col = _column(projections, "filename")
    findings_col = _column(reports, "findings")
    problems_col = _column(reports, "problems", "problem", "mesh")
    projection_col = None
    try:
        projection_col = _column(projections, "projection")
    except ValueError:
        if require_frontal:
            raise

    grouped: Dict[str, List[tuple[str, str]]] = {}
    for row in projections.to_dict(orient="records"):
        uid = _uid(row[p_uid])
        filename = _clean_text(row[filename_col])
        projection = _clean_text(row.get(projection_col, "")) if projection_col else ""
        if filename:
            grouped.setdefault(uid, []).append((filename, projection))

    manifest: List[dict] = []
    # mergesort is stable for the occasional duplicated UID in mirrors.
    for row in reports.sort_values(r_uid, kind="mergesort").to_dict(orient="records"):
        uid = _uid(row[r_uid])
        findings = _clean_text(row[findings_col])
        if require_findings and not findings:
            continue
        views = grouped.get(uid, [])
        if require_frontal:
            views = [(fn, pr) for fn, pr in views if "frontal" in pr.lower()]
        if not views:
            continue
        # Stable order makes cache use and split reproduction independent of CSV
        # row order.
        views = sorted(set(views), key=lambda item: (item[1].lower(), item[0]))
        filenames = [fn for fn, _ in views]
        labels = encode_problems(row.get(problems_col, ""))
        manifest.append({
            "uid": uid,
            "findings": findings,
            "problems": _clean_text(row.get(problems_col, "")),
            "filenames": filenames,
            "image_paths": [os.path.join(image_root, fn) for fn in filenames],
            "projections": [projection for _, projection in views],
            "label": labels,
        })
    return manifest


def manifest_stats(manifest: Sequence[Mapping[str, object]]) -> dict:
    labels = np.stack([np.asarray(item["label"], dtype=np.float32) for item in manifest]) \
        if manifest else np.zeros((0, len(CHEXPERT_LABELS)), dtype=np.float32)
    return {
        "n_samples": len(manifest),
        "has_both": sum(len(item.get("filenames", [])) >= 2 for item in manifest),
        "avg_labels": float(labels.sum(axis=1).mean()) if len(labels) else 0.0,
        "label_counts": labels.sum(axis=0).astype(int).tolist(),
    }


def split_train_test(
    manifest: Sequence[Mapping[str, object]],
    *,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    idx = np.arange(len(manifest))
    np.random.default_rng(seed).shuffle(idx)
    n_test = int(round(len(idx) * test_frac))
    return {
        "train": sorted(idx[n_test:].tolist()),
        "test": sorted(idx[:n_test].tolist()),
    }


def _primary_label(item: Mapping[str, object]) -> int:
    y = np.asarray(item["label"], dtype=np.float32)
    positives = np.flatnonzero(y > 0)
    return int(positives[0]) if len(positives) else 0


def partition_clients(
    manifest: Sequence[Mapping[str, object]],
    indices: Iterable[int],
    *,
    num_clients: int,
    scheme: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Partition indices deterministically, using primary-label Dirichlet skew."""

    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if scheme not in {"dirichlet", "iid"}:
        raise ValueError(f"unsupported partition scheme: {scheme}")
    if scheme == "dirichlet" and alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")

    rng = np.random.default_rng(seed)
    parts: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}
    idx = [int(i) for i in indices]
    if scheme == "iid":
        rng.shuffle(idx)
        for offset, sample_idx in enumerate(idx):
            parts[offset % num_clients].append(sample_idx)
    else:
        by_label: Dict[int, List[int]] = {}
        for sample_idx in idx:
            by_label.setdefault(_primary_label(manifest[sample_idx]), []).append(sample_idx)
        for label in sorted(by_label):
            label_idx = np.asarray(by_label[label], dtype=int)
            rng.shuffle(label_idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
            cuts = (np.cumsum(proportions)[:-1] * len(label_idx)).astype(int)
            for cid, chunk in enumerate(np.split(label_idx, cuts)):
                parts[cid].extend(int(i) for i in chunk)
    for values in parts.values():
        values.sort()
    return parts
