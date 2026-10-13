from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from face_reco.config import load_config
from face_reco.data.dataset import make_loader
from face_reco.device import resolve_device
from face_reco.metrics import (
    closed_set_accuracy,
    extract_embeddings,
    knn_accuracy,
    tpr_at_fpr,
)
from face_reco.models import ArcFaceModel, FaceClassifier, FaceEmbedder


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    loss = ckpt["loss"]
    num_classes = int(ckpt["num_classes"])
    cfg = ckpt.get("cfg", {})
    pretrained = None  # weights come from checkpoint
    if loss == "ce":
        model = FaceClassifier(num_classes, pretrained=pretrained, normalize=False)
    elif loss == "arcface":
        model = ArcFaceModel(
            num_classes,
            pretrained=pretrained,
            s=float(cfg.get("arcface_s", 64.0)),
            m=float(cfg.get("arcface_m", 0.5)),
        )
    elif loss == "triplet":
        model = FaceEmbedder(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown loss in checkpoint: {loss}")
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    allowed_extra = {
        "backbone.logits.weight",
        "backbone.logits.bias",
        "embedder.backbone.logits.weight",
        "embedder.backbone.logits.bias",
    }
    unexpected = set(incompatible.unexpected_keys) - allowed_extra
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Checkpoint does not match model: "
            f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model, ckpt


def evaluate(
    ckpt_path: str | Path,
    manifest_dir: str | Path,
    batch_size: int = 64,
    image_size: int = 160,
    num_workers: int = 2,
    num_pairs: int = 10000,
    fpr_targets: list[float] | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    loss = ckpt["loss"]
    manifest_dir = Path(manifest_dir)
    test_loader = make_loader(
        manifest_dir / "test.csv",
        batch_size,
        image_size,
        train=False,
        num_workers=num_workers,
        shuffle=False,
    )
    train_loader = make_loader(
        manifest_dir / "train.csv",
        batch_size,
        image_size,
        train=False,
        num_workers=num_workers,
        shuffle=False,
    )

    summary: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "loss": loss,
        "epoch": ckpt.get("epoch"),
        "device": str(device),
    }

    if loss in ("ce", "arcface"):
        summary["test_accuracy"] = closed_set_accuracy(
            model,
            test_loader,
            device,
            use_arcface_eval=(loss == "arcface"),
        )
    else:
        tr_emb, tr_y = extract_embeddings(model, train_loader, device)
        te_emb, te_y = extract_embeddings(model, test_loader, device)
        summary["test_knn_accuracy"] = knn_accuracy(tr_emb, tr_y, te_emb, te_y)

    emb_model = model.embedder if hasattr(model, "embedder") else model
    te_emb, te_y = extract_embeddings(emb_model, test_loader, device)
    summary["verification"] = tpr_at_fpr(
        te_emb,
        te_y,
        fpr_targets=fpr_targets or [0.001, 0.01, 0.1],
        num_pairs=num_pairs,
        seed=int(ckpt.get("cfg", {}).get("seed", 42)),
    )
    return summary


def eval_main_from_args(
    ckpt: str, config_paths: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if config_paths:
        cfg = load_config(*config_paths)
    manifest_dir = kwargs.get("manifest_dir") or cfg.get(
        "manifest_dir", "data/celeba/manifests"
    )
    return evaluate(
        ckpt,
        manifest_dir=manifest_dir,
        batch_size=int(kwargs.get("batch_size") or cfg.get("batch_size", 64)),
        image_size=int(kwargs.get("image_size") or cfg.get("image_size", 160)),
        num_workers=int(kwargs.get("num_workers") or cfg.get("num_workers", 2)),
        num_pairs=int(kwargs.get("num_pairs") or cfg.get("num_pairs", 10000)),
        fpr_targets=kwargs.get("fpr_targets") or cfg.get("fpr_targets"),
        device_name=str(kwargs.get("device") or cfg.get("device", "auto")),
    )
