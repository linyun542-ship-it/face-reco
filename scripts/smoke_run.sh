#!/usr/bin/env bash
# End-to-end smoke: prep small identity subset + train/eval all three losses.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-data/celeba}"
MANIFEST_DIR="${MANIFEST_DIR:-data/celeba/manifests_smoke}"
MAX_IDS="${MAX_IDS:-80}"
MAX_PER_ID="${MAX_PER_ID:-20}"

echo "==> Preparing smoke manifests (max_identities=${MAX_IDS})"
python scripts/prepare_splits.py \
  --data-root "$DATA_ROOT" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-identities "$MAX_IDS" \
  --max-images-per-identity "$MAX_PER_ID"

RESULTS_DIR="results/smoke"
mkdir -p "$RESULTS_DIR" /tmp/face-reco-smoke-overrides

for loss in ce triplet arcface; do
  override="/tmp/face-reco-smoke-overrides/${loss}.yaml"
  cat >"$override" <<EOF
output_dir: runs/smoke/${loss}
manifest_dir: ${MANIFEST_DIR}
EOF
  echo "==> Training smoke/${loss}"
  python scripts/train.py --config "configs/${loss}.yaml" configs/smoke.yaml "$override"
done

python - <<'PY'
import json
import shutil
from pathlib import Path

out = Path("results/smoke")
out.mkdir(parents=True, exist_ok=True)
combined = {}
for loss in ("ce", "triplet", "arcface"):
    summaries = sorted(
        Path(f"runs/smoke/{loss}").glob("*/summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not summaries:
        continue
    path = summaries[-1]
    data = json.loads(path.read_text())
    dest = out / f"{loss}_summary.json"
    shutil.copy2(path, dest)
    combined[loss] = data
(out / "combined.json").write_text(json.dumps(combined, indent=2))
slim = {
    k: {kk: vv for kk, vv in v.items() if kk != "data_meta"}
    for k, v in combined.items()
}
print(json.dumps(slim, indent=2))
PY

echo "==> Smoke summaries in ${RESULTS_DIR}"
