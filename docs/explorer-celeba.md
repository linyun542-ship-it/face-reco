# CelebA on NEU Explorer (scratch + Slurm GPU)

Cursor Cloud cannot write into your HPC account. Full training runs on **Explorer** (SSH or Open OnDemand terminal + Slurm GPU). Open OnDemand is only a login UI — you still must put CelebA on **cluster disk** (usually `/scratch/<NUID>`).

## 1. Get the code onto Explorer

```bash
ssh <you>@login.explorer.northeastern.edu
# pick a scratch project dir
export SCRATCH=/scratch/$USER          # adjust if your site uses /work/$USER
mkdir -p "$SCRATCH/face-reco"
cd "$SCRATCH/face-reco"
git clone <YOUR_GITHUB_REPO_URL> .
# or: rsync / scp a checkout from your laptop
```

Create a venv (or use a site Python module) and install:

```bash
module load python   # if required on Explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match cluster CUDA
pip install -e .
```

## 2. Download CelebA onto scratch

```bash
export SCRATCH=/scratch/$USER
bash scripts/download_celeba_explorer.sh
# equivalent:
# DATA_ROOT=$SCRATCH/datasets/celeba python scripts/download_celeba.py --data-root $DATA_ROOT
```

This pulls `img_align_celeba.zip` (~1.4 GB), extracts ~202K JPEGs, and fetches `identity_CelebA.txt` + `list_eval_partition.txt`.

**If Google Drive rate-limits:** download those three files on a machine that can reach Drive, then `scp`/`rsync` them to `$SCRATCH/datasets/celeba/` and re-run the script (existing files are skipped).

## 3. Build manifests

```bash
python scripts/prepare_splits.py \
  --data-root "$SCRATCH/datasets/celeba" \
  --manifest-dir "$SCRATCH/datasets/celeba/manifests"
```

Smoke subset (optional sanity check on a GPU node):

```bash
python scripts/prepare_splits.py \
  --data-root "$SCRATCH/datasets/celeba" \
  --manifest-dir "$SCRATCH/datasets/celeba/manifests_smoke" \
  --max-identities 80 --max-images-per-identity 20
```

## 4. Submit full-scale GPU jobs

```bash
mkdir -p logs
sbatch scripts/slurm_train.sh ce
sbatch scripts/slurm_train.sh arcface
sbatch scripts/slurm_train.sh triplet
squeue -u $USER
```

Edit `#SBATCH` lines in `scripts/slurm_train.sh` to match Explorer partitions (`sinfo`). Default configs target ~**91% CE** and ~**92.5% ArcFace** test accuracy after a full run; also inspect `verification` TPR@FPR in each run’s `summary.json`.

## Path checklist

| Item | Where |
|------|--------|
| Repo / code | `$SCRATCH/face-reco` |
| Images + labels | `$SCRATCH/datasets/celeba` |
| Manifests | `$SCRATCH/datasets/celeba/manifests` |
| Checkpoints / metrics | `$SCRATCH/face-reco/runs/<loss>/` |
| Slurm logs | `$SCRATCH/face-reco/logs/` |
