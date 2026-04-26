"""ArcFace and Sub-center ArcFace loss for metric learning.

Replaces Triplet Loss with angular margin classification, producing more
discriminative embeddings for few-shot keyword spotting.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin (ArcFace) loss.

    Maps L2-normalized embeddings to class logits via a cosine classifier
    with an additive angular margin penalty on the target class.

    Args:
        embedding_dim: Dimension of input embeddings.
        num_classes: Number of training classes.
        scale: Logit scaling factor (s).
        margin: Angular margin in radians (m).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss.

        Args:
            embeddings: (B, D) L2-normalized embeddings.
            labels: (B,) integer class labels.
        """
        normed_w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, normed_w)  # (B, C)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # Numerical safety: if cos(theta) < threshold, use linear penalty
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=normed_w.shape[0]).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.scale

        return F.cross_entropy(logits, labels)


class SubCenterArcFaceLoss(nn.Module):
    """Sub-center ArcFace (SCAF) loss.

    Each class has K learnable sub-centers; the maximum cosine similarity
    across sub-centers is used, making the loss robust to noisy/diverse
    intra-class distributions.

    Args:
        embedding_dim: Dimension of input embeddings.
        num_classes: Number of training classes.
        K: Number of sub-centers per class.
        scale: Logit scaling factor.
        margin: Angular margin in radians.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        K: int = 3,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.K = K
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.empty(num_classes * K, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Sub-center ArcFace loss.

        Args:
            embeddings: (B, D) L2-normalized embeddings.
            labels: (B,) integer class labels.
        """
        normed_w = F.normalize(self.weight, p=2, dim=1)  # (C*K, D)
        cosine_all = F.linear(embeddings, normed_w)       # (B, C*K)

        # Reshape to (B, C, K) and take max over sub-centers
        B = embeddings.shape[0]
        cosine_all = cosine_all.view(B, self.num_classes, self.K)
        cosine, _ = cosine_all.max(dim=2)  # (B, C)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.scale

        return F.cross_entropy(logits, labels)
