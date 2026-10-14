#!/usr/bin/env bash
#SBATCH --job-name=face-reco-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit using sbatch from the project root}"
cd "$ROOT"
[[ -f pyproject.toml ]] || { echo "Submit from the project root"; exit 1; }
if type module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE:-python/3.13.5}"
fi
PYTHON="${VENV_DIR:-${ROOT}/.venv-gpu}/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON"; exit 1; }
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/scratch/${USER}}/datasets/celeba}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/manifests}"
OUTPUT="runs/baseline/${SLURM_JOB_ID}/summary.json"
echo "[baseline] manifest_dir=${MANIFEST_DIR} output=${OUTPUT}"
"$PYTHON" -u scripts/eval_baseline.py \
  --manifest-dir "$MANIFEST_DIR" --device cuda \
  --num-workers 4 --num-pairs 10000 --seed 42 --output "$OUTPUT"
