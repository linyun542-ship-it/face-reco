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
# Install on a compute node; installation on the login node was killed.
srun --pty -p short --cpus-per-task=2 --mem=16G -t 01:00:00 /bin/bash
module load python/3.13.5
cd /scratch/$USER/face-reco
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.venv-gpu/bin/python -m pip install -e .
.venv-gpu/bin/python -c "import torch, facenet_pytorch, sklearn; print(torch.__version__, torch.version.cuda)"
# Only after both installs and the import check succeed:
exit
```

These pinned wheels target the observed V100 / driver 545.23.08 environment.
An existing `.venv-gpu` can be reused. Stop on errors: do not install with a
different interpreter if creating the environment fails. GPU availability is
checked by the training job on its allocated GPU node, not on the login node.

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
squeue -u $USER
```

Start with CE. Once it succeeds, submit ArcFace and triplet with the same script.
Create `logs/` **before** submitting: Slurm opens log files before the script runs.
Jobs continue after disconnecting. Use the actual job number printed by `sbatch`:

```bash
# Replace 123456 with the submitted job number.
tail -f logs/slurm-face-reco-123456.out
tail -50 logs/slurm-face-reco-123456.err
sacct -j 123456 --format=JobID,State,Elapsed,ExitCode
```

Ctrl+C exits log monitoring only. A disappearing queue entry does not prove
success: inspect `sacct`, errors, and `runs/ce/<timestamp>/summary.json`.
The script defaults to the observed `gpu` partition limit of eight hours.
This is a time limit, not an estimated runtime. If a run times out, existing best
checkpoints remain, but the current trainer does not automatically resume and
may not have produced final test metrics. Check the first epoch's timing before
assuming all 15 epochs will fit. Override site limits with `sbatch --time=...`
only on a partition where that duration is permitted.

The job uses `.venv-gpu/bin/python` explicitly. Set `VENV_DIR` to select another
environment and `PYTHON_MODULE` to select the module that created it.

Edit `#SBATCH` lines in `scripts/slurm_train.sh` to match Explorer partitions (`sinfo`). The ~**91% CE** and ~**92.5% ArcFace** figures are project goals, not verified full-run results. Report only measured metrics from `summary.json`, with the split, number of identities, epochs, and verification pair count. Smoke results demonstrate execution only; they do not establish full-data accuracy.

## Path checklist

| Item | Where |
|------|--------|
| Repo / code | `$SCRATCH/face-reco` |
| Images + labels | `$SCRATCH/datasets/celeba` |
| Manifests | `$SCRATCH/datasets/celeba/manifests` |
| Checkpoints / metrics | `$SCRATCH/face-reco/runs/<loss>/` |
| Slurm logs | `$SCRATCH/face-reco/logs/` |
