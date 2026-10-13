# Face Reco — CelebA Fine-Tune (CE / Triplet / ArcFace)

Fine-tune a **VGGFace2-pretrained InceptionResNetV1** (`facenet-pytorch`) on **CelebA**, comparing three losses:

| Loss | Closed-set metric | Full-scale target |
|------|-------------------|-------------------|
| Cross-entropy | Test accuracy | ~**91%** |
| ArcFace | Test accuracy | ~**92.5%** |
| Triplet | Nearest-centroid (kNN) accuracy | Report + TPR@FPR |

All runs also log verification **TPR@FPR** (FPR ∈ {1e-3, 1e-2, 1e-1}) on cosine similarity of L2-normalized embeddings.

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
# Full-scale recipes (GPU recommended; matches resume-scale targets)
python scripts/train.py --config configs/ce.yaml
python scripts/train.py --config configs/triplet.yaml
python scripts/train.py --config configs/arcface.yaml

# Smoke overlay
python scripts/train.py --config configs/ce.yaml configs/smoke.yaml
```

Checkpoints and `summary.json` land under `runs/<loss>/<timestamp>/`.

### Full-scale knobs (resume targets)

| Setting | Value |
|---------|-------|
| Backbone | InceptionResNetV1, `pretrained=vggface2` |
| Input | 160×160, flip aug |
| Epochs | 15 |
| Batch | 64 (triplet: 16 ids × 4 images) |
| LR | 1e-4 backbone / 1e-3 head; freeze backbone 1 epoch (CE/ArcFace) |
| ArcFace | s=64, m=0.5 |
| Triplet margin | 0.5 (batch-hard) |

CelebA's official split is identity-disjoint, so closed-set test accuracy is not applicable there; use ROC AUC and TPR@FPR. The ~91% CE and ~92.5% ArcFace figures are **unverified project goals** for a separate per-identity closed-set split. For interviews, cite actual metrics and name the split protocol. See the [Explorer guide](docs/explorer-celeba.md) for the persistent GPU environment and eight-hour job limit.

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
- **No GPU in the reference smoke environment**: numbers below resume targets are expected on the 80-identity / 1-epoch subset.
- Triplet has no native softmax accuracy; we report nearest-centroid accuracy on embeddings plus the same TPR@FPR protocol.
- Prefer hosting on **GitHub**; if `gh` is not authenticated in the agent environment, push the branch from your laptop (`git remote add` + PR).
