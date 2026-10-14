# Face Verification on CelebA

A reproducible comparison of cross-entropy, ArcFace, and batch-hard triplet fine-tuning for **unseen-identity face verification**, using a VGGFace2-pretrained InceptionResNetV1 in PyTorch and Slurm GPU jobs.

## Measured results

All three completed runs use 15 epochs and 10,000 test pairs. Checkpoints are selected by validation ROC AUC.

| Training objective | Test ROC AUC | TPR at 0.1% FPR | TPR at 1% FPR | TPR at 10% FPR |
|---|---:|---:|---:|---:|
| Cross-entropy | 0.98027000 | 40.42% | 74.04% | 95.08% |
| ArcFace | 0.98277232 | **85.44%** | **91.44%** | 95.84% |
| Triplet | **0.98290528** | 47.72% | 83.64% | **96.60%** |

**Takeaway:** ArcFace performs best at the stricter false-acceptance operating points. Triplet has a slightly higher AUC, but the difference from ArcFace is only 0.00013296 and has not been assessed for statistical significance.

These are verification metrics, not classification accuracy. Training contains 8,156 identities and 162,734 images; validation contains 980 identities and 19,862 images; testing contains 997 identities and 19,959 images. Recorded train/validation and train/test identity overlaps are zero. The test identities are unseen during CelebA fine-tuning; overlap with VGGFace2 pretraining identities has not been audited.

## Project contents

- Training, evaluation, CelebA preparation, and Slurm submission scripts.
- [Recorded full-run metrics and comparison](results/full/comparison.md), including per-epoch logs and configurations. Heavy checkpoints and face images are excluded from Git.
- [Detailed experiment report (Chinese)](docs/experiment-report-zh.md).
- A two-image cosine similarity demo and an unmodified pretrained baseline evaluator.

## Reproduce the comparison

Follow the [Explorer setup guide](docs/explorer-celeba.md) to download data, prepare official manifests and submit one job per loss. Keep the same manifests and seed for all experiments.

```bash
mkdir -p logs
sbatch scripts/slurm_train.sh ce
sbatch scripts/slurm_train.sh arcface
sbatch scripts/slurm_train.sh triplet
```

Regenerate the recorded result comparison without model dependencies:

```bash
python scripts/compare_results.py results/full/ce/*/summary.json results/full/arcface/*/summary.json results/full/triplet/*/summary.json --output results/full/comparison.md
```

## Two-face demonstration

Use your locally backed-up checkpoint and two already cropped/aligned face images. This script does not detect or align faces automatically.

```bash
python scripts/verify_pair.py --checkpoint runs/arcface/20261013T185644Z/best.pt --image-a face_a.jpg --image-b face_b.jpg
```

The output is cosine similarity, **not a probability**. No same-person decision is produced unless you provide a validation-calibrated `--threshold`. Load only checkpoints you trust. This demo is an engineering addition; real-image inference has not yet been verified in the published experiment.

## Pretrained baseline (next experiment)

Submit from the Explorer project directory using the same environment and manifests as the completed runs (create logs before submission):

```bash
mkdir -p logs
sbatch scripts/slurm_baseline.sh
# Results: runs/baseline/<job-id>/summary.json
# Logs: logs/slurm-face-reco-baseline-<job-id>.out
```

Baseline evaluation is implemented but **no baseline score is reported yet**. First-time pretrained weights may require network access.

## Interpretation and limitations

CE and ArcFace reached their best validation AUC at epoch 2; Triplet at epoch 14. Continuing to reduce training loss did not consistently improve validation performance. Cross-objective loss magnitudes are not comparable.

This is a single-seed comparison of three training recipes, not an isolated causal study of loss functions: freeze schedules and sampling differ. Manifest paths and metadata match, but the archive does not contain manifest hashes or the actual sampled pair lists. Low-FPR estimates have finite-pair uncertainty; bootstrap intervals, multiple seeds, pretraining identity-overlap checks, and a baseline remain future work. Test ROC working points are not calibrated deployment thresholds.

CelebA data is not distributed here. Review its original dataset terms before use. Checkpoints remain in the author's local backup rather than being published as Git blobs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Requires Python 3.10+, PyTorch, and ~2–3 GB disk for aligned CelebA (zip + extract).

## NEU Explorer (full training)

Full CelebA training is meant for **Explorer Slurm GPU**, not the Cursor VM and not a login-node self-hosted worker.

- **Open OnDemand** is only a browser login UI — it does **not** download CelebA for you.
- Put data on **cluster scratch** (`/scratch/$USER/...`), then submit GPU jobs with `sbatch`.

```bash
ssh <you>@login.explorer.northeastern.edu
export SCRATCH=/scratch/$USER
cd $SCRATCH/face-reco   # clone this repo here first
bash scripts/download_celeba_explorer.sh
python scripts/prepare_splits.py \
  --data-root $SCRATCH/datasets/celeba \
  --manifest-dir $SCRATCH/datasets/celeba/manifests
mkdir -p logs
sbatch scripts/slurm_train.sh ce
sbatch scripts/slurm_train.sh arcface
sbatch scripts/slurm_train.sh triplet
```

Step-by-step: [`docs/explorer-celeba.md`](docs/explorer-celeba.md).

## Data (local / Cursor smoke)

```bash
# Downloads img_align_celeba.zip + identity/partition files via gdown
python scripts/download_celeba.py --data-root data/celeba

# Official identity-disjoint manifests for open-set verification
python scripts/prepare_splits.py --data-root data/celeba --manifest-dir data/celeba/manifests

# Per-identity split for closed-set classification accuracy
python scripts/prepare_splits.py \
  --data-root data/celeba \
  --manifest-dir data/celeba/manifests_closed_set \
  --split-strategy stratified
```

If Google Drive rate-limits the zip, place `img_align_celeba.zip`, `identity_CelebA.txt`, and `list_eval_partition.txt` under the data root and re-run the download script (it skips existing files) then prepare.

### Smoke subset (CPU / limited VM)

```bash
python scripts/prepare_splits.py \
  --data-root data/celeba \
  --manifest-dir data/celeba/manifests_smoke \
  --max-identities 80 \
  --max-images-per-identity 20
```

Or one-shot (prep + train all three losses):

```bash
bash scripts/smoke_run.sh
```

## Train

Configs live in `configs/`. Later YAML files override earlier keys.

```bash
# Full-scale recipes (GPU recommended)
python scripts/train.py --config configs/ce.yaml
python scripts/train.py --config configs/triplet.yaml
python scripts/train.py --config configs/arcface.yaml

# Smoke overlay
python scripts/train.py --config configs/ce.yaml configs/smoke.yaml
```

Checkpoints and `summary.json` land under `runs/<loss>/<timestamp>/`.

### Full-scale training settings

| Setting | Value |
|---------|-------|
| Backbone | InceptionResNetV1, `pretrained=vggface2` |
| Input | 160×160, flip aug |
| Epochs | 15 |
| Batch | 64 (triplet: 16 ids × 4 images) |
| LR | 1e-4 backbone / 1e-3 head; freeze backbone 1 epoch (CE/ArcFace) |
| ArcFace | s=64, m=0.5 |
| Triplet margin | 0.5 (batch-hard) |

CelebA's official split is identity-disjoint, so closed-set test accuracy is not applicable there; use ROC AUC and TPR@FPR. See measured verification results above and the [Explorer guide](docs/explorer-celeba.md) for setup.

## Eval

```bash
python scripts/eval.py \
  --checkpoint runs/ce/<timestamp>/best.pt \
  --config configs/ce.yaml
```

## Metrics in `summary.json`

- `test_accuracy` (CE / ArcFace) or `test_knn_accuracy` (triplet)
- `verification`: `tpr@fpr=0.001`, `tpr@fpr=0.01`, `tpr@fpr=0.1`, `roc_auc`

Smoke results (small identity subset, 1 epoch, CPU) are illustrative only — see `results/smoke/` after `smoke_run.sh`.

## Layout

```
configs/           # ce, triplet, arcface, smoke
docs/              # Explorer / scratch download guide
src/face_reco/     # data, models, losses, train, eval
scripts/           # download*, prepare, train, eval, smoke_run, slurm_train
results/smoke/     # copied smoke summaries
```

## Notes

- **Full training target = NEU Explorer GPU** via `scripts/slurm_train.sh`. Cursor smoke validates the pipeline only.
- Smoke runs validate execution on a small subset; use the recorded full runs for measured results.
- Triplet has no native softmax accuracy; we report nearest-centroid accuracy on embeddings plus the same TPR@FPR protocol.
- Prefer hosting on **GitHub**; if `gh` is not authenticated in the agent environment, push the branch from your laptop (`git remote add` + PR).
