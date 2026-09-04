"""MIMIC-IV + MIMIC-CXR benchmark utilities.

This package intentionally separates benchmark validation from the later QFL
method. It implements access/cohort audits, complementarity Gate 0 evaluation,
and reproducible FL partition construction.
"""

from .access import audit_access
from .cohort import audit_cohort
from .gate import evaluate_complementarity_gate
from .partition import build_fl_partition

__all__ = [
    "audit_access",
    "audit_cohort",
    "evaluate_complementarity_gate",
    "build_fl_partition",
]
