"""Download CelebA aligned images + identity / partition annotations."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import gdown
from tqdm import tqdm

# Public Google Drive file IDs commonly used for CelebA aligned faces + annotations.
# If Drive rate-limits, place files manually under data_root and re-run.
# IDs aligned with torchvision.datasets.celeba file list.
CELEBA_FILES = {
    "img_align_celeba.zip": {
        "id": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    },
    "list_eval_partition.txt": {
        "id": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    },
    "identity_CelebA.txt": {
        "id": "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
    },
}


def _download_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return
    print(f"[download] {dest.name} …")
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True, retries=3)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"Failed to download {dest.name}. "
            "Place the file manually in the data root and re-run."
        )


def _extract_images(zip_path: Path, out_dir: Path) -> Path:
    target = out_dir / "img_align_celeba"
    if target.is_dir() and any(target.glob("*.jpg")):
        n = sum(1 for _ in target.glob("*.jpg"))
        print(f"[skip] images already extracted ({n:,} jpgs in {target})")
        return target

    print(f"[extract] {zip_path} → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="extract"):
            zf.extract(member, out_dir)

    # Zip may nest img_align_celeba/ or dump jpgs at root.
    if target.is_dir():
        return target
    jpgs = list(out_dir.glob("*.jpg"))
    if jpgs:
        target.mkdir(parents=True, exist_ok=True)
        for p in jpgs:
            shutil.move(str(p), str(target / p.name))
        return target
    raise RuntimeError(f"Could not find extracted images under {out_dir}")


def download_celeba(data_root: str | Path, skip_images: bool = False) -> Path:
    """
    Download CelebA assets into data_root.

    Layout:
      data_root/
        img_align_celeba.zip
        img_align_celeba/*.jpg
        identity_CelebA.txt
        list_eval_partition.txt
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    for name, meta in CELEBA_FILES.items():
        if skip_images and name.endswith(".zip"):
            continue
        dest = root / name
        # Allow user-provided copies with alternate names.
        if name == "identity_CelebA.txt" and not dest.exists():
            alt = root / "identity_CelebA.txt"
            if alt.exists():
                continue
        _download_file(meta["id"], dest)

    if not skip_images:
        _extract_images(root / "img_align_celeba.zip", root)

    return root
