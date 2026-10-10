from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from face_reco.models.backbone import FaceEmbedder


class ArcMarginProduct(nn.Module):
    """Additive angular margin (ArcFace) head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.clamp(cosine**2, 0, 1)).clamp(min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcFaceModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: str | None = "vggface2",
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.embedder = FaceEmbedder(pretrained=pretrained)
        self.head = ArcMarginProduct(
            self.embedder.embedding_dim, num_classes, s=s, m=m
        )

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        emb = self.embedder(x, normalize=True)
        logits = None
        if labels is not None:
            logits = self.head(emb, labels)
        return logits, emb
