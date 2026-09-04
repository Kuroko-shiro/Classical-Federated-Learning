#!/usr/bin/env bash
set -euo pipefail

# MedMod-compatible MIMIC acquisition.
# Password is never stored: wget prompts interactively via --ask-password.
# Requires a PhysioNet credentialed account with access to both datasets.

PHYSIONET_USER="${PHYSIONET_USER:-}"
DATA_ROOT="${DATA_ROOT:-$PWD/data/mimic}"
MIMIC_IV_VERSION="${MIMIC_IV_VERSION:-1.0}"
MIMIC_CXR_JPG_VERSION="${MIMIC_CXR_JPG_VERSION:-2.1.0}"

if [[ -z "$PHYSIONET_USER" ]]; then
  echo "ERROR: set PHYSIONET_USER to your PhysioNet username." >&2
  echo "Example: export PHYSIONET_USER='your_username'" >&2
  exit 2
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget is required. On macOS: brew install wget" >&2
  exit 2
fi

IV_DIR="$DATA_ROOT/mimiciv/$MIMIC_IV_VERSION"
CXR_DIR="$DATA_ROOT/mimic-cxr-jpg/$MIMIC_CXR_JPG_VERSION"
mkdir -p "$IV_DIR" "$CXR_DIR"

MIMIC_IV_BASE="https://physionet.org/files/mimiciv/$MIMIC_IV_VERSION/"
CXR_BASE="https://physionet.org/files/mimic-cxr-jpg/$MIMIC_CXR_JPG_VERSION/"

cat <<EOF
Downloading protected MIMIC data.
  PhysioNet user: $PHYSIONET_USER
  MIMIC-IV:       v$MIMIC_IV_VERSION -> $IV_DIR
  MIMIC-CXR-JPG:  v$MIMIC_CXR_JPG_VERSION metadata only -> $CXR_DIR

MedMod's current MIMIC-IV extractor expects the v1.x core/ layout.
CXR images are NOT downloaded at this stage; they will be subsetted after cohort linkage.
EOF

# 1) MIMIC-IV v1.0: download the benchmark source CSV tree.
# -N and -c make reruns resumable/idempotent.
(
  cd "$IV_DIR"
  wget -r -N -c -np -nH --cut-dirs=2 \
    --user "$PHYSIONET_USER" --ask-password \
    "$MIMIC_IV_BASE"
)

# 2) CXR metadata required by MedMod/linkage. v2.1.0 retains the v2.0.0 tables.
CXR_FILES=(
  "mimic-cxr-2.0.0-metadata.csv.gz"
  "mimic-cxr-2.0.0-split.csv.gz"
  "mimic-cxr-2.0.0-chexpert.csv.gz"
  "mimic-cxr-2.0.0-negbio.csv.gz"
  "IMAGE_FILENAMES"
  "RECORDS"
)

for f in "${CXR_FILES[@]}"; do
  (
    cd "$CXR_DIR"
    wget -N -c --user "$PHYSIONET_USER" --ask-password "${CXR_BASE}${f}"
  )
done

# MedMod reads uncompressed CSVs. Keep .gz originals and materialize CSV copies.
for gz in "$CXR_DIR"/*.csv.gz; do
  [[ -e "$gz" ]] || continue
  csv="${gz%.gz}"
  if [[ ! -f "$csv" || "$gz" -nt "$csv" ]]; then
    gzip -cd "$gz" > "$csv"
  fi
done

# MIMIC-IV v1.0 releases may be compressed. Materialize CSVs while keeping originals.
while IFS= read -r -d '' gz; do
  csv="${gz%.gz}"
  if [[ ! -f "$csv" || "$gz" -nt "$csv" ]]; then
    gzip -cd "$gz" > "$csv"
  fi
done < <(find "$IV_DIR" -type f -name '*.csv.gz' -print0)

echo
echo "Download/preparation finished. Run:"
echo "python scripts/mimic_benchmark.py audit-access \\\"
echo "  --mimic-iv-root '$IV_DIR' \\\"
echo "  --mimic-cxr-root '$CXR_DIR' \\\"
echo "  --output-dir results/mimic_benchmark/access"
