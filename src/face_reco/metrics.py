from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        # Models return (logits|None, emb) or just emb for FaceEmbedder
        out = model(x) if not hasattr(model, "embedder") else _forward_emb(model, x)
        if isinstance(out, tuple):
            emb = out[1]
        else:
            emb = out
        emb = F.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu())
        labels.append(y)
        if max_batches is not None and i + 1 >= max_batches:
            break
    return torch.cat(embs).numpy(), torch.cat(labels).numpy()


def _forward_emb(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "embedder"):
        return model.embedder(x, normalize=True)
    return model(x)


def sample_pairs(
    labels: np.ndarray,
    num_pairs: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx_a, idx_b, is_same) with ~50/50 genuine/impostor."""
    rng = np.random.default_rng(seed)
    by_id: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        by_id.setdefault(int(lab), []).append(i)

    ids_multi = [i for i, idxs in by_id.items() if len(idxs) >= 2]
    all_ids = list(by_id.keys())
    if not ids_multi or len(all_ids) < 2:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    n_pos = num_pairs // 2
    n_neg = num_pairs - n_pos
    a_idx, b_idx, y = [], [], []

    for _ in range(n_pos):
        ident = int(rng.choice(ids_multi))
        pair = rng.choice(by_id[ident], size=2, replace=False)
        a_idx.append(int(pair[0]))
        b_idx.append(int(pair[1]))
        y.append(1)

    for _ in range(n_neg):
        i1, i2 = rng.choice(all_ids, size=2, replace=False)
        a_idx.append(int(rng.choice(by_id[int(i1)])))
        b_idx.append(int(rng.choice(by_id[int(i2)])))
        y.append(0)

    return np.array(a_idx), np.array(b_idx), np.array(y)


def tpr_at_fpr(
    embeddings: np.ndarray,
    labels: np.ndarray,
    fpr_targets: Iterable[float] = (1e-3, 1e-2, 1e-1),
    num_pairs: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    a, b, y_true = sample_pairs(labels, num_pairs=num_pairs, seed=seed)
    if y_true.size == 0:
        out = {f"tpr@fpr={target:g}": 0.0 for target in fpr_targets}
        out["roc_auc"] = 0.0
        out["num_pairs"] = 0.0
        return out
    # Cosine similarity for L2-normalized embeddings == dot product
    scores = np.sum(embeddings[a] * embeddings[b], axis=1)
    fpr, tpr, _ = roc_curve(y_true, scores)
    out: dict[str, float] = {}
    for target in fpr_targets:
        # Highest TPR with FPR <= target
        mask = fpr <= target
        val = float(tpr[mask].max()) if mask.any() else 0.0
        out[f"tpr@fpr={target:g}"] = val
    # AUC approximation via trapz on ROC
    order = np.argsort(fpr)
    out["roc_auc"] = float(np.trapz(tpr[order], fpr[order]))
    out["num_pairs"] = float(num_pairs)
    return out


@torch.no_grad()
def closed_set_accuracy(
    model,
    loader,
    device: torch.device,
    use_arcface_eval: bool = False,
) -> float:
    """
    Classification accuracy.

    For ArcFace, evaluate with cosine to class weight vectors (no margin) when
    labels are unavailable for the margin head — here we still have labels, so
    we use cosine logits: s * cos(theta).
    """
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if use_arcface_eval and hasattr(model, "head") and hasattr(model.head, "weight"):
            emb = model.embedder(x, normalize=True)
            w = F.normalize(model.head.weight, dim=1)
            logits = emb @ w.T
        else:
            out = model(x) if not callable(getattr(model, "forward", None)) else model(x)
            if isinstance(out, tuple):
                logits = out[0]
                if logits is None:
                    emb = out[1]
                    # Fallback: should not happen for CE
                    raise RuntimeError("Model returned no logits for accuracy")
            else:
                logits = out
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def knn_accuracy(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    test_emb: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """Nearest-centroid accuracy in embedding space (for triplet probe)."""
    classes = np.unique(train_y)
    centroids = []
    for c in classes:
        centroids.append(train_emb[train_y == c].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    test_n = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-9)
    sims = test_n @ centroids.T
    pred = classes[sims.argmax(axis=1)]
    return float((pred == test_y).mean())
