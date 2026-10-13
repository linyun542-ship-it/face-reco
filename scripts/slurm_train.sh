#!/usr/bin/env bash
#SBATCH --job-name=face-reco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

# Match the Python used to create the persistent GPU environment.
if type module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE:-python/3.13.5}"
fi
VENV_DIR="${VENV_DIR:-${ROOT}/.venv-gpu}"
PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] Missing GPU environment: $PYTHON"
  echo "See docs/explorer-celeba.md for installation on a compute node."
  exit 1
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/scratch/${USER}}/datasets/celeba}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/manifests}"

for FILE in train.csv val.csv test.csv meta.json; do
if [[ ! -s "${MANIFEST_DIR}/${FILE}" ]]; then
  echo "[error] Missing or empty ${MANIFEST_DIR}/${FILE}"
  echo "Run download + prepare on Explorer scratch first (see README / download_celeba_explorer.sh)."
  exit 1
fi
done

CONFIG="configs/${LOSS}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] Unknown loss '${LOSS}'. Use: ce | triplet | arcface"
  exit 1
fi

# Fail before loading data if the selected environment cannot use this GPU.
"$PYTHON" -u - <<'PY'
import sys
import torch
import facenet_pytorch
import sklearn
print(f"[environment] python={sys.executable} torch={torch.__version__} cuda={torch.version.cuda}")
if torch.version.cuda is None or not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable. Install GPU PyTorch and submit with --gres=gpu:1.")
x = torch.ones(1, device="cuda")
print(f"[environment] gpu={torch.cuda.get_device_name(0)} check={(x + x).item()}")
PY

# Write a tiny override so jobs read cluster data paths.
OVERRIDE="$(mktemp /tmp/face-reco-slurm-XXXXXX.yaml)"
trap 'rm -f "$OVERRIDE"' EXIT
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

"$PYTHON" -u scripts/train.py --config "${CONFIG}" "${OVERRIDE}"
