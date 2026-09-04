from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def _find_first(root: Path | None, candidates: list[str]) -> str | None:
    if root is None or not root.exists():
        return None
    for name in candidates:
        p = root / name
        if p.exists():
            return str(p)
    return None


def _du_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    total = 0
    try:
        for base, _, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(base) / f).stat().st_size
                except OSError:
                    pass
        return total
    except OSError:
        return None


def audit_access(mimic_iv_root, mimic_cxr_root, output_dir, benchmark_repo=None) -> dict[str, Any]:
    """Filesystem-only audit of the protected MIMIC prerequisites."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    iv = Path(mimic_iv_root).expanduser().resolve() if mimic_iv_root else None
    cxr = Path(mimic_cxr_root).expanduser().resolve() if mimic_cxr_root else None
    repo = Path(benchmark_repo).expanduser().resolve() if benchmark_repo else None

    # MedMod's current extractor is written against the MIMIC-IV v1.x schema,
    # where patients/admissions live in core/. MIMIC-IV >=2.0 moved them to hosp/.
    patients = _find_first(
        iv,
        [
            "core/patients.csv",
            "core/patients.csv.gz",
            "hosp/patients.csv",
            "hosp/patients.csv.gz",
            "patients.csv",
            "patients.csv.gz",
        ],
    )
    admissions = _find_first(
        iv,
        [
            "core/admissions.csv",
            "core/admissions.csv.gz",
            "hosp/admissions.csv",
            "hosp/admissions.csv.gz",
            "admissions.csv",
            "admissions.csv.gz",
        ],
    )
    metadata = _find_first(
        cxr,
        [
            "mimic-cxr-2.0.0-metadata.csv",
            "mimic-cxr-2.0.0-metadata.csv.gz",
            "mimic-cxr-2.1.0-metadata.csv",
            "mimic-cxr-2.1.0-metadata.csv.gz",
            "metadata.csv",
            "metadata.csv.gz",
        ],
    )
    audit = {
        "mimic_iv_available": bool(iv and iv.exists()),
        "mimic_cxr_available": bool(cxr and cxr.exists()),
        "mimic_iv_root": str(iv) if iv else None,
        "mimic_cxr_root": str(cxr) if cxr else None,
        "mimic_iv_patients_table": patients,
        "mimic_iv_admissions_table": admissions,
        "mimic_cxr_metadata_table": metadata,
        "subject_id_linkage_prerequisites_present": bool(patients and metadata),
        "physionet_access_status": "unknown_from_local_audit",
        "mimic_iv_disk_bytes": _du_bytes(iv),
        "mimic_cxr_disk_bytes": _du_bytes(cxr),
        "benchmark_repo_present": bool(repo and repo.exists()),
        "benchmark_repo_path": str(repo) if repo else None,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "git_available": shutil.which("git") is not None,
    }
    audit["ready_for_cohort_reproduction"] = bool(
        audit["mimic_iv_available"]
        and audit["mimic_cxr_available"]
        and audit["subject_id_linkage_prerequisites_present"]
    )
    (out / "mimic_access_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# MIMIC Access Audit",
        "",
        f"- MIMIC-IV locally available: **{audit['mimic_iv_available']}**",
        f"- MIMIC-CXR locally available: **{audit['mimic_cxr_available']}**",
        f"- subject_id linkage prerequisites present: **{audit['subject_id_linkage_prerequisites_present']}**",
        f"- ready for cohort reproduction: **{audit['ready_for_cohort_reproduction']}**",
        f"- MIMIC-IV root: `{audit['mimic_iv_root']}`",
        f"- MIMIC-CXR root: `{audit['mimic_cxr_root']}`",
        f"- patients table: `{patients}`",
        f"- admissions table: `{admissions}`",
        f"- CXR metadata: `{metadata}`",
        "",
        "PhysioNet credential status is not inferred from filesystem state.",
    ]
    (out / "mimic_access_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit
