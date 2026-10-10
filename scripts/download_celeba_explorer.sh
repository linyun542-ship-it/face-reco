#!/usr/bin/env bash
# Download + extract CelebA onto NEU Explorer cluster scratch.
#
# Open OnDemand is only a login UI — it does NOT put data on disk for you.
# Run this over SSH (or an OnDemand terminal / interactive job) so files land
# on shared scratch that Slurm GPU jobs can read.
#
# Typical usage on Explorer:
#   ssh <you>@login.explorer.northeastern.edu
#   cd /scratch/<NUID>/face-reco   # or clone the repo there
#   bash scripts/download_celeba_explorer.sh
#
# Override paths:
#   SCRATCH=/scratch/$USER DATA_ROOT=/scratch/$USER/datasets/celeba \
#     bash scripts/download_celeba_explorer.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer SCRATCH; fall back to common Explorer-style locations, then ./data.
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    DATA_ROOT="${SCRATCH}/datasets/celeba"
  elif [[ -d "/scratch/${USER:-}" ]]; then
    DATA_ROOT="/scratch/${USER}/datasets/celeba"
  elif [[ -d "/work/${USER:-}" ]]; then
    DATA_ROOT="/work/${USER}/datasets/celeba"
  else
    DATA_ROOT="${ROOT}/data/celeba"
    echo "[warn] No SCRATCH/USER scratch dir found; using ${DATA_ROOT}"
  fi
fi

mkdir -p "${DATA_ROOT}"
echo "[info] CelebA data root: ${DATA_ROOT}"
echo "[info] Need ~2–3 GB free (zip + extracted aligned faces)."
df -h "$(dirname "${DATA_ROOT}")" || true

# Use project venv if present; otherwise system/module python.
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "[error] python3 not found. On Explorer: module load python/<ver> then pip install -e ."
  exit 1
fi

# Ensure gdown is importable when running outside an installed package env.
if ! "${PYTHON}" -c "import gdown" 2>/dev/null; then
  echo "[info] Installing gdown for ${PYTHON} …"
  "${PYTHON}" -m pip install --user -q gdown || "${PYTHON}" -m pip install -q gdown
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON}" "${ROOT}/scripts/download_celeba.py" --data-root "${DATA_ROOT}"

# Point the repo's default data/celeba at scratch via symlink if still empty.
LOCAL_LINK="${ROOT}/data/celeba"
if [[ ! -e "${LOCAL_LINK}" ]]; then
  mkdir -p "${ROOT}/data"
  ln -sfn "${DATA_ROOT}" "${LOCAL_LINK}"
  echo "[info] Symlinked ${LOCAL_LINK} -> ${DATA_ROOT}"
elif [[ -L "${LOCAL_LINK}" ]]; then
  ln -sfn "${DATA_ROOT}" "${LOCAL_LINK}"
  echo "[info] Updated symlink ${LOCAL_LINK} -> ${DATA_ROOT}"
else
  echo "[info] ${LOCAL_LINK} already exists; manifests should use --data-root ${DATA_ROOT}"
fi

echo
echo "Next on Explorer:"
echo "  ${PYTHON} scripts/prepare_splits.py --data-root ${DATA_ROOT} --manifest-dir ${DATA_ROOT}/manifests"
echo "  sbatch scripts/slurm_train.sh ce     # or triplet / arcface"
echo
echo "If Google Drive rate-limits, copy these onto ${DATA_ROOT} manually then re-run:"
echo "  img_align_celeba.zip  identity_CelebA.txt  list_eval_partition.txt"
