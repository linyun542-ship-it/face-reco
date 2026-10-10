#!/usr/bin/env bash
#SBATCH --job-name=face-reco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Full-scale CelebA fine-tune on NEU Explorer (Slurm GPU).
# Open OnDemand → Clusters → Explorer Shell Access (or SSH), then:
#
#   cd /scratch/<NUID>/face-reco
#   bash scripts/download_celeba_explorer.sh
#   python scripts/prepare_splits.py \
#     --data-root "${SCRATCH:-/scratch/$USER}/datasets/celeba" \
#     --manifest-dir "${SCRATCH:-/scratch/$USER}/datasets/celeba/manifests"
#   mkdir -p logs
#   sbatch scripts/slurm_train.sh ce
#   sbatch scripts/slurm_train.sh arcface
#   sbatch scripts/slurm_train.sh triplet
#
# Adjust --partition / --gres to match `sinfo` on Explorer.
set -euo pipefail

LOSS="${1:-ce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs runs

# Load site modules if present (names vary; ignore failures).
module load cuda 2>/dev/null || true
module load python 2>/dev/null || true

if [[ -x "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/scratch/${USER}}/datasets/celeba}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/manifests}"

if [[ ! -f "${MANIFEST_DIR}/train.csv" ]]; then
  echo "[error] Missing ${MANIFEST_DIR}/train.csv"
  echo "Run download + prepare on Explorer scratch first (see README / download_celeba_explorer.sh)."
  exit 1
fi

CONFIG="configs/${LOSS}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] Unknown loss '${LOSS}'. Use: ce | triplet | arcface"
  exit 1
fi

# Write a tiny override so jobs read cluster data paths.
OVERRIDE="$(mktemp /tmp/face-reco-slurm-XXXXXX.yaml)"
cat >"${OVERRIDE}" <<EOF
data_root: ${DATA_ROOT}
manifest_dir: ${MANIFEST_DIR}
device: cuda
num_workers: 4
output_dir: runs/${LOSS}
EOF

echo "[slurm] host=$(hostname) cuda=$(command -v nvidia-smi || true)"
nvidia-smi || true
echo "[slurm] loss=${LOSS} data_root=${DATA_ROOT} manifest_dir=${MANIFEST_DIR}"

python scripts/train.py --config "${CONFIG}" "${OVERRIDE}"
rm -f "${OVERRIDE}"
