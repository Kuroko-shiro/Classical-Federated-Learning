#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV=${VENV:-.venv-vitaldb}
OUT=${OUT:-artifacts/vitaldb_stage_ab}

if [ ! -d "$VENV" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$(dirname "$0")/requirements.txt"

python "$(dirname "$0")/stage_ab_audit.py" --out "$OUT"
python "$(dirname "$0")/stage_b_smoke_download.py" --audit-dir "$OUT" --n-cases 3
python "$(dirname "$0")/stage_c_header_qc.py" --audit-dir "$OUT" --n-cases 10 --interval 1.0

echo
echo "Stage A/B + smoke QC complete: $OUT"
