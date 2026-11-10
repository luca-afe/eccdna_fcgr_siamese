"""
Siamese Network architecture (V3 with GroupNorm), Euclidean Contrastive Loss,
dynamic semi-hard negative mining, and downstream evaluation routines.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FCGRCNNEncoderV3(nn.Module):
    """
    CNN Encoder condiviso della Rete Siamese (versione aggiornata V3 con GroupNorm
    e AdaptiveAvgPool2d, compatibile con input multi-canale e risoluzioni variabili).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        z = self.embedding_head(x)
        return F.normalize(z, p=2, dim=1, eps=1e-8)


class SiameseV3(nn.Module):
    """
    Rete Neurale Siamese (V3) che concatena il batch `(x1, x2)` in un singolo
    forward pass sull'encoder condiviso `FCGRCNNEncoderV3`.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = FCGRCNNEncoderV3(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x1.shape[0]
        x = torch.cat([x1, x2], dim=0)
        z = self.encoder(x)
        return z[:batch_size], z[batch_size:]


class EuclideanContrastiveLoss(nn.Module):
    """
    Contrastive Loss basata su distanza Euclidea tra embedding L2-normalizzati:
        L = y * d^2 + (1 - y) * max(0, margin - d)^2
    """

    def __init__(self, margin: float = 1.25) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = z1.float()
        z2 = z2.float()
        target = target.float()

        distances = torch.linalg.vector_norm(z1 - z2, ord=2, dim=1)
        positive_loss = target * distances.pow(2)
        negative_loss = (1.0 - target) * F.relu(self.margin - distances).pow(2)
        loss = (positive_loss + negative_loss).mean()
        return loss, distances


# Alias retrocompatibili
FCGRCNNEncoder = FCGRCNNEncoderV3
SiameseNetwork = SiameseV3
ContrastiveLoss = EuclideanContrastiveLoss


@torch.no_grad()
def select_semihard_negatives(
    model: SiameseV3,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
    amp_enabled: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Seleziona dinamicamente per ciascun anchor il candidato negativo semi-hard
    (d_pos < d_neg < margin) più vicino all'anchor, con fallback sicuro.
    """
    model.eval()
    device = anchor.device
    batch_size = anchor.shape[0]
    n_candidates = negatives.shape[1]

    negatives_flat = negatives.reshape(
        batch_size * n_candidates, *negatives.shape[2:]
    )
    all_images = torch.cat([anchor, positive, negatives_flat], dim=0)

    with torch.autocast(
        device_type=device.type,
        dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16),
        enabled=amp_enabled,
    ):
        all_embeddings = model.encoder(all_images)

    z_anchor = all_embeddings[:batch_size].float()
    z_positive = all_embeddings[batch_size : 2 * batch_size].float()
    z_negatives = (
        all_embeddings[2 * batch_size :]
        .float()
        .reshape(batch_size, n_candidates, -1)
    )

    d_pos = torch.linalg.vector_norm(z_anchor - z_positive, ord=2, dim=1)
    d_neg = torch.linalg.vector_norm(
        z_anchor[:, None, :] - z_negatives, ord=2, dim=2
    )

    semihard_mask = (d_neg > d_pos[:, None]) & (d_neg < margin)
    has_semihard = semihard_mask.any(dim=1)

    masked_distances = torch.where(
        semihard_mask, d_neg, torch.full_like(d_neg, float("inf"))
    )
    semihard_indices = masked_distances.argmin(dim=1)

    safer_mask = d_neg > d_pos[:, None]
    safer_distances = torch.where(
        safer_mask, d_neg, torch.full_like(d_neg, float("inf"))
    )
    safer_indices = safer_distances.argmin(dim=1)
    has_safer = safer_mask.any(dim=1)
    least_hard_indices = d_neg.argmax(dim=1)

    fallback_indices = torch.where(has_safer, safer_indices, least_hard_indices)
    selected_indices = torch.where(has_semihard, semihard_indices, fallback_indices)

    batch_indices = torch.arange(batch_size, device=negatives.device)
    selected_negatives = negatives[batch_indices, selected_indices]
    selected_distances = d_neg[batch_indices, selected_indices]

    stats = {
        "semihard_fraction": has_semihard.float().mean().item(),
        "fallback_fraction": (~has_semihard).float().mean().item(),
        "mean_d_pos": d_pos.mean().item(),
        "mean_selected_d_neg": selected_distances.mean().item(),
        "selected_valid_fraction": (selected_distances > d_pos).float().mean().item(),
        "selected_inside_margin_fraction": (
            selected_distances < margin
        ).float().mean().item(),
    }
    return selected_negatives, stats


@torch.no_grad()
def evaluate_pairwise(
    model: SiameseV3,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = True,
) -> Dict[str, float]:
    """
    Valuta il modello Siamese su un DataLoader di coppie restituendo
    loss, ROC-AUC, d+, d-, gap e d-prime.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_distances = []

    for batch in loader:
        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16),
            enabled=amp_enabled,
        ):
            z1, z2 = model(x1, x2)

        loss, distances = criterion(z1, z2, target)
        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(target.cpu().numpy())
        all_distances.append(distances.cpu().numpy())

    targets = np.concatenate(all_targets)
    distances = np.concatenate(all_distances)

    positive_distances = distances[targets == 1]
    negative_distances = distances[targets == 0]

    d_pos = float(positive_distances.mean())
    d_neg = float(negative_distances.mean())
    gap = d_neg - d_pos
    pooled_variance = 0.5 * (positive_distances.var() + negative_distances.var())
    d_prime = float(gap / np.sqrt(pooled_variance + 1e-12))
    auc = float(roc_auc_score(targets, -distances))

    return {
        "loss": total_loss / total_samples,
        "auc": auc,
        "d_pos": d_pos,
        "d_neg": d_neg,
        "gap": gap,
        "d_prime": d_prime,
    }


@torch.no_grad()
def extract_embeddings(
    model: SiameseV3,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae gli embedding L2-normalizzati e le label da un SingleFCGRDataset loader.
    """
    model.eval()
    embeddings_all = []
    labels_all = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16),
            enabled=amp_enabled,
        ):
            z = model.encoder(x)
        embeddings_all.append(z.float().cpu().numpy())
        labels_all.append(batch["label"].numpy())

    return np.concatenate(embeddings_all, axis=0), np.concatenate(labels_all, axis=0)


def compute_topk_similarity_matrix(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    n_classes: int,
    top_k: int = 20,
    query_batch_size: int = 512,
) -> np.ndarray:
    """
    Calcola la matrice di similarità coseno media sui top-k vicini per ciascuna classe
    rispetto alla banca di riferimento (Reference Bank).
    """
    result = np.zeros((len(query_embeddings), n_classes), dtype=np.float32)
    class_reference_embeddings = {
        class_id: reference_embeddings[reference_labels == class_id]
        for class_id in range(n_classes)
    }

    for start in range(0, len(query_embeddings), query_batch_size):
        end = min(start + query_batch_size, len(query_embeddings))
        query_batch = query_embeddings[start:end]

        for class_id in range(n_classes):
            ref = class_reference_embeddings[class_id]
            similarities = query_batch @ ref.T
            top_k_values = np.partition(
                similarities,
                kth=(similarities.shape[1] - top_k),
                axis=1,
            )[:, -top_k:]
            result[start:end, class_id] = top_k_values.mean(axis=1)

    return result

