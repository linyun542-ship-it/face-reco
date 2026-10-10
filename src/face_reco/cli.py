from __future__ import annotations

import argparse
import json

from face_reco.config import load_config
from face_reco.data.download import download_celeba
from face_reco.data.prepare import prepare_manifests
from face_reco.eval import eval_main_from_args
from face_reco.train import train_main_from_args


def download_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download CelebA assets")
    p.add_argument("--data-root", default="data/celeba")
    p.add_argument("--skip-images", action="store_true")
    args = p.parse_args(argv)
    download_celeba(args.data_root, skip_images=args.skip_images)


def prepare_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build CelebA train/val/test manifests")
    p.add_argument("--data-root", default="data/celeba")
    p.add_argument("--manifest-dir", default="data/celeba/manifests")
    p.add_argument("--max-identities", type=int, default=None)
    p.add_argument("--max-images-per-identity", type=int, default=None)
    p.add_argument("--min-images-per-identity", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    prepare_manifests(
        args.data_root,
        manifest_dir=args.manifest_dir,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        min_images_per_identity=args.min_images_per_identity,
        seed=args.seed,
    )


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fine-tune face recognition model")
    p.add_argument(
        "--config",
        required=True,
        nargs="+",
        help="YAML config(s); later files override earlier keys",
    )
    args = p.parse_args(argv)
    summary = train_main_from_args(args.config)
    print(json.dumps({"ok": True, "run_dir": summary.get("run_dir")}, indent=2))


def eval_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", nargs="*", default=None)
    p.add_argument("--manifest-dir", default=None)
    args = p.parse_args(argv)
    summary = eval_main_from_args(
        args.checkpoint,
        config_paths=args.config,
        manifest_dir=args.manifest_dir,
    )
    print(json.dumps(summary, indent=2))
    out = Path_safe(args.checkpoint)
    if out:
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def Path_safe(ckpt: str):
    from pathlib import Path

    parent = Path(ckpt).resolve().parent
    return parent / "eval_summary.json"
