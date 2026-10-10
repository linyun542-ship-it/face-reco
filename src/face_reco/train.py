from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from face_reco.config import ensure_dir, load_config
from face_reco.data.dataset import make_loader, make_triplet_loader
from face_reco.device import resolve_device
from face_reco.losses import BatchHardTripletLoss
from face_reco.metrics import (
    closed_set_accuracy,
    extract_embeddings,
    knn_accuracy,
    tpr_at_fpr,
)
from face_reco.models import ArcFaceModel, FaceClassifier, FaceEmbedder


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(str(cfg.get("device", "auto")))
    loss_name = cfg["loss"]
    manifest_dir = Path(cfg["manifest_dir"])
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"
    meta = json.loads((manifest_dir / "meta.json").read_text(encoding="utf-8"))
    num_classes = int(meta["num_identities"])

    run_dir = ensure_dir(
        Path(cfg.get("output_dir", f"runs/{loss_name}"))
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"

    image_size = int(cfg.get("image_size", 160))
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 2))
    epochs = int(cfg.get("epochs", 10))
    pretrained = cfg.get("pretrained", "vggface2")

    print(f"[train] loss={loss_name} device={device} classes={num_classes} run={run_dir}")

    if loss_name == "ce":
        model: nn.Module = FaceClassifier(
            num_classes, pretrained=pretrained, normalize=False
        )
        criterion: nn.Module = nn.CrossEntropyLoss()
        train_loader = make_loader(
            train_csv, batch_size, image_size, True, num_workers
        )
    elif loss_name == "arcface":
        model = ArcFaceModel(
            num_classes,
            pretrained=pretrained,
            s=float(cfg.get("arcface_s", 64.0)),
            m=float(cfg.get("arcface_m", 0.5)),
        )
        criterion = nn.CrossEntropyLoss()
        train_loader = make_loader(
            train_csv, batch_size, image_size, True, num_workers
        )
    elif loss_name == "triplet":
        model = FaceEmbedder(pretrained=pretrained)
        criterion = BatchHardTripletLoss(margin=float(cfg.get("triplet_margin", 0.5)))
        train_loader = make_triplet_loader(
            train_csv,
            image_size=image_size,
            identities_per_batch=int(cfg.get("identities_per_batch", 16)),
            images_per_identity=int(cfg.get("images_per_identity", 4)),
            num_workers=num_workers,
            seed=int(cfg.get("seed", 42)),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    model = model.to(device)

    # Optimizer with optional higher head LR
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name or "fc" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    lr = float(cfg.get("lr", 1e-4))
    head_lr = float(cfg.get("head_lr", lr))
    param_groups = [{"params": backbone_params, "lr": lr}]
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    optimizer = Adam(param_groups, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    freeze_epochs = int(cfg.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0 and hasattr(model, "embedder"):
        model.embedder.freeze()
        print(f"[train] freezing backbone for {freeze_epochs} epoch(s)")

    val_loader = make_loader(
        val_csv, batch_size, image_size, False, num_workers, shuffle=False
    )
    test_loader = make_loader(
        test_csv, batch_size, image_size, False, num_workers, shuffle=False
    )

    best_score = -1.0
    best_path = run_dir / "best.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if (
            freeze_epochs > 0
            and epoch == freeze_epochs + 1
            and hasattr(model, "embedder")
        ):
            model.embedder.unfreeze()
            print("[train] unfroze backbone")

        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if loss_name == "ce":
                logits, _ = model(x)
                loss = criterion(logits, y)
            elif loss_name == "arcface":
                logits, _ = model(x, y)
                loss = criterion(logits, y)
            else:
                emb = model(x, normalize=True)
                loss = criterion(emb, y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        train_loss = running_loss / max(n_batches, 1)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": time.time() - t0,
        }

        # Validation
        if loss_name in ("ce", "arcface"):
            val_acc = closed_set_accuracy(
                model,
                val_loader,
                device,
                use_arcface_eval=(loss_name == "arcface"),
            )
            record["val_accuracy"] = val_acc
            score = val_acc
        else:
            train_eval_loader = make_loader(
                train_csv, batch_size, image_size, False, num_workers, shuffle=False
            )
            tr_emb, tr_y = extract_embeddings(
                model, train_eval_loader, device, max_batches=50
            )
            va_emb, va_y = extract_embeddings(model, val_loader, device)
            val_acc = knn_accuracy(tr_emb, tr_y, va_emb, va_y)
            record["val_knn_accuracy"] = val_acc
            score = val_acc

        history.append(record)
        _append_jsonl(metrics_path, record)
        print(f"[epoch {epoch}] {record}")

        if score >= best_score:
            best_score = score
            _save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "loss": loss_name,
                    "num_classes": num_classes,
                    "cfg": cfg,
                    "score": best_score,
                },
            )

    # Final test evaluation with best checkpoint
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    summary: dict[str, Any] = {
        "loss": loss_name,
        "device": str(device),
        "num_classes": num_classes,
        "best_val_score": best_score,
        "epochs": epochs,
        "run_dir": str(run_dir),
        "data_meta": meta,
    }

    if loss_name in ("ce", "arcface"):
        test_acc = closed_set_accuracy(
            model,
            test_loader,
            device,
            use_arcface_eval=(loss_name == "arcface"),
        )
        summary["test_accuracy"] = test_acc
    else:
        train_eval_loader = make_loader(
            train_csv, batch_size, image_size, False, num_workers, shuffle=False
        )
        tr_emb, tr_y = extract_embeddings(model, train_eval_loader, device)
        te_emb, te_y = extract_embeddings(model, test_loader, device)
        summary["test_knn_accuracy"] = knn_accuracy(tr_emb, tr_y, te_emb, te_y)

    # Verification metrics on test embeddings
    emb_model = model.embedder if hasattr(model, "embedder") else model
    te_emb, te_y = extract_embeddings(emb_model, test_loader, device)
    summary["verification"] = tpr_at_fpr(
        te_emb,
        te_y,
        fpr_targets=cfg.get("fpr_targets", [0.001, 0.01, 0.1]),
        num_pairs=int(cfg.get("num_pairs", 10000)),
        seed=int(cfg.get("seed", 42)),
    )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def train_main_from_args(config_paths: list[str]) -> dict[str, Any]:
    cfg = load_config(*config_paths)
    return train_from_config(cfg)
