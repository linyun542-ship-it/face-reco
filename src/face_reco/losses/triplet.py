from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss.

    Expects embeddings already L2-normalized (or not — uses pairwise Euclidean).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise squared distances
        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.unsqueeze(1)
        same = labels.eq(labels.T).bool()
        diff = ~same

        # Hardest positive: max dist among same-class (exclude self)
        pos_dist = dist.clone()
        pos_dist[~same] = -1.0
        # diagonal should not win
        pos_dist.fill_diagonal_(-1.0)
        hardest_pos, _ = pos_dist.max(dim=1)

        # Hardest negative: min dist among different-class
        neg_dist = dist.clone()
        neg_dist[~diff] = float("inf")
        hardest_neg, _ = neg_dist.min(dim=1)

        # Skip anchors with no valid positive/negative
        valid = (hardest_pos >= 0) & torch.isfinite(hardest_neg)
        if valid.sum() == 0:
            return embeddings.sum() * 0.0

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return loss.mean()
