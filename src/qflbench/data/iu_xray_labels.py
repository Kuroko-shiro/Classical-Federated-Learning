"""Map IU X-ray ``Problems`` terms to the CheXpert 14-label space.

The public IU X-ray CSV contains free-form, semicolon-delimited problem terms
rather than a fixed target vector.  This module keeps the mapping deterministic,
case-insensitive and independent of torch so the frozen split can be reproduced
in a lightweight environment.

This is a weak-label mapping, not a CheXpert labeler replacement.  The exact
label order is part of the experiment protocol and must not be changed without
regenerating the frozen split and all checkpoints.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np


CHEXPERT_LABELS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


_PATTERNS = {
    "No Finding": (
        r"\bnormal\b", r"\bno (?:acute )?(?:cardiopulmonary )?(?:finding|disease)\b",
        r"\bclear lungs?\b",
    ),
    "Enlarged Cardiomediastinum": (
        r"\benlarged cardiomediast", r"\bmediastinal widen", r"\bwidened mediast",
    ),
    "Cardiomegaly": (r"\bcardiomegal", r"\benlarged (?:cardiac|heart)",),
    "Lung Opacity": (
        r"\bopacit", r"\binfiltrat", r"\bairspace disease", r"\bairspace process",
    ),
    "Lung Lesion": (
        r"\bnodule", r"\bmass(?:es)?\b", r"\bgranuloma", r"\blung lesion",
    ),
    "Edema": (r"\bedema", r"\bvascular congestion", r"\bpulmonary congestion",),
    "Consolidation": (r"\bconsolidat",),
    "Pneumonia": (r"\bpneumonia", r"\bpneumonitis",),
    "Atelectasis": (r"\batelecta",),
    "Pneumothorax": (r"\bpneumothorax",),
    "Pleural Effusion": (r"\bpleural effusion", r"\bcostophrenic .*blunt",),
    "Pleural Other": (
        r"\bpleural thick", r"\bpleural plaque", r"\bpleural scar",
        r"\bfibrothorax", r"\bhemothorax",
    ),
    "Fracture": (r"\bfracture", r"\bcompression deform",),
    "Support Devices": (
        r"\bcatheter", r"\bcentral line", r"\bpicc\b", r"\btube\b",
        r"\bpacemaker", r"\bdefibrillator", r"\bprosthes", r"\bhardware\b",
        r"\bstent\b", r"\bclips?\b",
    ),
}

_COMPILED = {
    label: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for label, patterns in _PATTERNS.items()
}


def _normalise_terms(problems: object) -> str:
    if problems is None:
        return ""
    try:
        if bool(np.isnan(problems)):  # pandas/numpy missing scalar
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(problems, str):
        return " ".join(problems.replace("|", ";").split())
    if isinstance(problems, Iterable):
        return "; ".join(str(term) for term in problems)
    return str(problems)


def encode_problems(problems: object) -> np.ndarray:
    """Return a float32 multi-hot vector in :data:`CHEXPERT_LABELS` order.

    ``No Finding`` is mutually exclusive with positive abnormality labels.  An
    unrecognised non-empty problem string is left as an all-zero vector rather
    than silently being treated as normal.
    """

    text = _normalise_terms(problems)
    out = np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
    for i, label in enumerate(CHEXPERT_LABELS):
        if any(pattern.search(text) for pattern in _COMPILED[label]):
            out[i] = 1.0
    if out[1:].any():
        out[0] = 0.0
    return out


def active_labels(problems: object) -> List[str]:
    """Human-readable labels, useful for manifest audits and unit tests."""

    encoded = encode_problems(problems)
    return [label for label, flag in zip(CHEXPERT_LABELS, encoded) if flag]
