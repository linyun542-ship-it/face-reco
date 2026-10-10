from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceEmbedder(nn.Module):
    """InceptionResNetV1 backbone producing L2-normalized 512-D embeddings."""

    def __init__(self, pretrained: str | None = "vggface2", classify: bool = False):
        super().__init__()
        # classify=False → returns embeddings (512-D for vggface2)
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)
        self.embedding_dim = 512

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        emb = self.backbone(x)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class FaceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: str | None = "vggface2",
        normalize: bool = False,
    ):
        super().__init__()
        self.embedder = FaceEmbedder(pretrained=pretrained)
        self.head = ClassifierHead(self.embedder.embedding_dim, num_classes)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedder(x, normalize=self.normalize)
        logits = self.head(emb)
        return logits, emb
