"""Build train/val/test manifests from CelebA identity + partition files."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_identity(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = int(parts[1])
    return mapping


def _read_partition(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = int(parts[1])
    return mapping


def _remap_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map original CelebA identity ids to contiguous [0, N)."""
    unique = sorted({r["identity"] for r in rows})
    remap = {old: new for new, old in enumerate(unique)}
    out = []
    for r in rows:
        out.append({**r, "identity": remap[r["identity"]], "celeba_id": r["identity"]})
    return out


def prepare_manifests(
    data_root: str | Path,
    manifest_dir: str | Path | None = None,
    max_identities: int | None = None,
    max_images_per_identity: int | None = None,
    min_images_per_identity: int = 2,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Write CSV + JSON manifests for train (0), val (1), test (2).

    Smoke mode: keep only the first `max_identities` identities (by frequency)
    and optionally cap images per identity.
    """
    root = Path(data_root)
    img_dir = root / "img_align_celeba"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    identity_path = root / "identity_CelebA.txt"
    partition_path = root / "list_eval_partition.txt"
    if not identity_path.exists() or not partition_path.exists():
        raise FileNotFoundError(
            "Need identity_CelebA.txt and list_eval_partition.txt under data_root"
        )

    identities = _read_identity(identity_path)
    partitions = _read_partition(partition_path)

    by_id: dict[int, list[str]] = defaultdict(list)
    for fname, ident in identities.items():
        if fname not in partitions:
            continue
        if not (img_dir / fname).exists():
            continue
        by_id[ident].append(fname)

    # Drop sparse identities.
    by_id = {i: fns for i, fns in by_id.items() if len(fns) >= min_images_per_identity}

    if max_identities is not None:
        ranked = sorted(by_id.items(), key=lambda kv: len(kv[1]), reverse=True)
        by_id = dict(ranked[:max_identities])

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for ident, fnames in by_id.items():
        fnames = list(fnames)
        rng.shuffle(fnames)
        if max_images_per_identity is not None:
            fnames = fnames[:max_images_per_identity]
        for fname in fnames:
            rows.append(
                {
                    "filename": fname,
                    "path": str(img_dir / fname),
                    "identity": ident,
                    "split": partitions[fname],  # 0 train, 1 val, 2 test
                }
            )

    rows = _remap_labels(rows)
    num_identities = len({r["identity"] for r in rows})

    out_dir = Path(manifest_dir) if manifest_dir else root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = {0: "train", 1: "val", 2: "test"}
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {
        name: sum(1 for r in rows if r["split"] == split_id)
        for split_id, name in split_names.items()
    }

    # Subset filters often leave official val/test too thin for verification.
    # Rebuild a stratified 80/10/10 when subsetting or when any split is empty.
    use_rebalance = (
        max_identities is not None
        or max_images_per_identity is not None
        or any(counts[n] == 0 for n in split_names.values())
    )
    if use_rebalance:
        print("[prepare] Using per-identity 80/10/10 split for subset/rebalance")
        by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_label[r["identity"]].append(r)
        rebuilt: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
        for _ident, items in by_label.items():
            items = list(items)
            rng.shuffle(items)
            n = len(items)
            if n == 1:
                rebuilt[0].append({**items[0], "split": 0})
                continue
            if n == 2:
                rebuilt[0].append({**items[0], "split": 0})
                rebuilt[2].append({**items[1], "split": 2})
                continue
            n_train = max(1, int(0.8 * n))
            n_val = max(1, int(0.1 * n))
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)
            for i, item in enumerate(items):
                if i < n_train:
                    s = 0
                elif i < n_train + n_val:
                    s = 1
                else:
                    s = 2
                rebuilt[s].append({**item, "split": s})
        rows_by_split = rebuilt
    else:
        rows_by_split = {
            split_id: [r for r in rows if r["split"] == split_id]
            for split_id in split_names
        }

    for split_id, name in split_names.items():
        split_rows = rows_by_split[split_id]
        counts[name] = len(split_rows)
        csv_path = out_dir / f"{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "path", "identity", "celeba_id", "split"]
            )
            writer.writeheader()
            writer.writerows(split_rows)
        paths[name] = csv_path

    meta = {
        "num_identities": num_identities,
        "counts": counts,
        "max_identities": max_identities,
        "max_images_per_identity": max_images_per_identity,
        "image_dir": str(img_dir),
        "seed": seed,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    print(json.dumps(meta, indent=2))
    return paths
