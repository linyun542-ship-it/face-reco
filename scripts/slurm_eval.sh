#!/usr/bin/env bash
#SBATCH --job-name=face-reco-eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

CHECKPOINT="${1:?Usage: sbatch scripts/slurm_eval.sh runs/<loss>/<timestamp>/best.pt}"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ ! -f "${ROOT}/pyproject.toml" ]]; then
  echo "[error] Submit from the face-reco repository root"
  exit 1
fi
cd "$ROOT"

if type module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE:-python/3.13.5}"
fi
PYTHON="${VENV_DIR:-${ROOT}/.venv-gpu}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] Missing GPU environment: $PYTHON"
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/scratch/${USER}}/datasets/celeba}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/manifests}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[error] Missing checkpoint: $CHECKPOINT"
  exit 1
fi

LOSS="$("$PYTHON" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["loss"])' "$CHECKPOINT")"
CONFIG="configs/${LOSS}.yaml"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[eval] checkpoint=$CHECKPOINT loss=$LOSS manifests=$MANIFEST_DIR"
"$PYTHON" -u scripts/eval.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --manifest-dir "$MANIFEST_DIR"
