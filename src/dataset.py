"""
PyTorch Dataset classes for eccDNA FCGR Siamese Network training and evaluation.

Includes:
- `SiamesePairDataset` (on-the-fly sequence -> FCGR pair generation with LRU cache)
- `SiamesePairDatasetMemmap` (memmap-backed pair generation)
- `RandomSiamesePairDataset` (static random pairs for deterministic validation/testing)
- `BalancedSiamesePairDataset` (class-balanced batches for random warm-up)
- `BalancedSemiHardCandidateDataset` (class-balanced anchors with K negative candidates for dynamic semi-hard mining)
- `SingleFCGRDataset` (single FCGR extraction for reference-bank and test embedding evaluation)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fcgr import sequence_to_fcgr_tensor


def _format_fcgr_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Ensure the returned FCGR array has shape `(C, H, W)` as a `torch.float32` tensor.
    Supports both 2D `(H, W)` single-channel and 3D `(C, H, W)` multi-channel FCGRs.
    """
    tensor = torch.from_numpy(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


class SiamesePairDataset(Dataset):
    """
    Dataset che genera coppie (positive/negative) calcolando la FCGR on-demand
    dalle sequenze nucleotidiche con cache LRU in memoria.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        sequences: Dict[str, str],
        k: int = 6,
        pairs_per_epoch: int = 50_000,
        positive_probability: float = 0.5,
        seed: int = 42,
        deterministic: bool = False,
        cache_size: int = 4096,
        encoder_fn: Optional[Callable[[str, int], Tuple[torch.Tensor, int]]] = None,
    ) -> None:
        self.metadata = metadata[["id", "class_id", "disease_clean"]].copy()
        self.metadata["id"] = self.metadata["id"].astype(str)

        self.sequences = sequences
        self.k = int(k)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.positive_probability = float(positive_probability)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.encoder_fn = encoder_fn or sequence_to_fcgr_tensor

        self.class_to_ids: Dict[int, np.ndarray] = {}
        for class_id, group in self.metadata.groupby("class_id"):
            self.class_to_ids[int(class_id)] = group["id"].astype(str).to_numpy()

        self.classes = np.array(sorted(self.class_to_ids.keys()), dtype=np.int64)

        for class_id, ids in self.class_to_ids.items():
            if len(ids) < 2:
                raise ValueError(f"La classe {class_id} ha meno di 2 sample.")

        self.rng = np.random.default_rng(self.seed)
        self._cached_fcgr = lru_cache(maxsize=cache_size)(self._compute_fcgr_tensor)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _compute_fcgr_tensor(self, sample_id: str) -> torch.Tensor:
        sequence = self.sequences[sample_id]
        tensor, n_valid = self.encoder_fn(sequence, self.k)
        if n_valid == 0:
            raise ValueError(f"Nessun k-mer valido per sample {sample_id}")
        return tensor

    def _sample_pair_ids(self, rng: np.random.Generator):
        class1 = int(rng.choice(self.classes))
        ids_class1 = self.class_to_ids[class1]
        id1 = str(rng.choice(ids_class1))

        is_positive = rng.random() < self.positive_probability
        if is_positive:
            class2 = class1
            id2 = id1
            while id2 == id1:
                id2 = str(rng.choice(ids_class1))
            target = 1.0
        else:
            possible_classes = self.classes[self.classes != class1]
            class2 = int(rng.choice(possible_classes))
            ids_class2 = self.class_to_ids[class2]
            id2 = str(rng.choice(ids_class2))
            target = 0.0

        return id1, id2, class1, class2, target

    def get_pair_info(self, index: int) -> Dict[str, object]:
        rng = np.random.default_rng(self.seed + index)
        id1, id2, class1, class2, target = self._sample_pair_ids(rng)
        return {
            "id1": id1,
            "id2": id2,
            "class1": class1,
            "class2": class2,
            "target": target,
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        rng = (
            np.random.default_rng(self.seed + index)
            if self.deterministic
            else self.rng
        )
        id1, id2, class1, class2, target = self._sample_pair_ids(rng)
        fcgr1 = self._cached_fcgr(id1)
        fcgr2 = self._cached_fcgr(id2)

        return {
            "x1": fcgr1,
            "x2": fcgr2,
            "target": torch.tensor(target, dtype=torch.float32),
            "id1": id1,
            "id2": id2,
            "class1": class1,
            "class2": class2,
        }


class SiamesePairDatasetMemmap(SiamesePairDataset):
    """
    Versione memory-mapped del SiamesePairDataset: legge le FCGR precomputate da disco.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        fcgr_memmap: np.ndarray,
        id_to_row: Dict[str, int],
        k: int = 6,
        pairs_per_epoch: int = 50_000,
        positive_probability: float = 0.5,
        seed: int = 42,
        deterministic: bool = False,
    ) -> None:
        self.fcgr_memmap = fcgr_memmap
        self.id_to_row = id_to_row

        super().__init__(
            metadata=metadata,
            sequences={},
            k=k,
            pairs_per_epoch=pairs_per_epoch,
            positive_probability=positive_probability,
            seed=seed,
            deterministic=deterministic,
            cache_size=0,
        )

    def _compute_fcgr_tensor(self, sample_id: str) -> torch.Tensor:
        row = self.id_to_row[str(sample_id)]
        fcgr = np.array(self.fcgr_memmap[row], dtype=np.float32, copy=True)
        return _format_fcgr_tensor(fcgr)


class RandomSiamesePairDataset(Dataset):
    """
    Coppie statiche pre-campionate con seed fisso per la validazione pairwise (Notebook 14 / 04).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        fcgr_memmap: np.ndarray,
        id_to_row: Dict[str, int],
        n_pairs: int,
        positive_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.metadata = metadata.copy().reset_index(drop=True)
        self.fcgr_memmap = fcgr_memmap
        self.n_pairs = int(n_pairs)
        self.positive_fraction = float(positive_fraction)
        self.seed = int(seed)

        self.rows = (
            self.metadata["id"].astype(str).map(id_to_row).to_numpy(dtype=np.int64)
        )
        self.labels = self.metadata["class_id"].to_numpy(dtype=np.int64)
        self.classes = np.array(sorted(np.unique(self.labels)), dtype=np.int64)
        self.class_to_indices = {
            int(c): np.where(self.labels == c)[0] for c in self.classes
        }

        self._generate_pairs()

    def _generate_pairs(self) -> None:
        rng = np.random.default_rng(self.seed)
        n_positive = int(self.n_pairs * self.positive_fraction)

        targets = np.zeros(self.n_pairs, dtype=np.float32)
        targets[:n_positive] = 1.0
        rng.shuffle(targets)

        anchors = rng.integers(0, len(self.labels), size=self.n_pairs)
        partners = np.empty(self.n_pairs, dtype=np.int64)

        for i, anchor_idx in enumerate(anchors):
            anchor_idx = int(anchor_idx)
            anchor_class = int(self.labels[anchor_idx])

            if targets[i] == 1.0:
                candidates = self.class_to_indices[anchor_class]
                partner_idx = anchor_idx
                while partner_idx == anchor_idx:
                    partner_idx = int(rng.choice(candidates))
            else:
                negative_classes = self.classes[self.classes != anchor_class]
                negative_class = int(rng.choice(negative_classes))
                partner_idx = int(
                    rng.choice(self.class_to_indices[negative_class])
                )

            partners[i] = partner_idx

        self.row1 = self.rows[anchors]
        self.row2 = self.rows[partners]
        self.targets = targets

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x1 = np.array(self.fcgr_memmap[int(self.row1[index])], dtype=np.float32, copy=True)
        x2 = np.array(self.fcgr_memmap[int(self.row2[index])], dtype=np.float32, copy=True)
        return {
            "x1": _format_fcgr_tensor(x1),
            "x2": _format_fcgr_tensor(x2),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }


class BalancedSiamesePairDataset(Dataset):
    """
    Dataset di coppie con batch bilanciati per classe (es. 5 anchor per ciascuna delle 12 classi
    in ogni batch da 60) utilizzato nella fase di Random Warm-Up.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        fcgr_memmap: np.ndarray,
        id_to_row: Dict[str, int],
        n_batches: int,
        batch_size: int = 60,
        positive_fraction: float = 0.50,
        seed: int = 42,
    ) -> None:
        self.metadata = metadata.copy().reset_index(drop=True)
        self.fcgr_memmap = fcgr_memmap
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.seed = int(seed)

        self.rows = (
            self.metadata["id"].astype(str).map(id_to_row).to_numpy(dtype=np.int64)
        )
        self.labels = self.metadata["class_id"].to_numpy(dtype=np.int64)
        self.classes = np.array(sorted(np.unique(self.labels)), dtype=np.int64)
        self.n_classes = len(self.classes)

        if self.batch_size % self.n_classes != 0:
            raise ValueError("batch_size must be divisible by number of classes")

        self.samples_per_class_per_batch = self.batch_size // self.n_classes
        self.class_to_indices = {
            int(c): np.where(self.labels == c)[0] for c in self.classes
        }
        self.n_pairs = self.n_batches * self.batch_size
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch) * 100_003)
        row1, row2, targets_all, anchor_classes_all = [], [], [], []
        positives_per_batch = int(round(self.batch_size * self.positive_fraction))

        for _ in range(self.n_batches):
            anchor_classes = np.repeat(
                self.classes, self.samples_per_class_per_batch
            )
            rng.shuffle(anchor_classes)

            targets = np.zeros(self.batch_size, dtype=np.float32)
            targets[:positives_per_batch] = 1.0
            rng.shuffle(targets)

            for anchor_class, target in zip(anchor_classes, targets):
                anchor_class = int(anchor_class)
                anchor_idx = int(rng.choice(self.class_to_indices[anchor_class]))

                if target == 1.0:
                    candidates = self.class_to_indices[anchor_class]
                    partner_idx = anchor_idx
                    while partner_idx == anchor_idx:
                        partner_idx = int(rng.choice(candidates))
                else:
                    negative_classes = self.classes[self.classes != anchor_class]
                    negative_class = int(rng.choice(negative_classes))
                    partner_idx = int(
                        rng.choice(self.class_to_indices[negative_class])
                    )

                row1.append(int(self.rows[anchor_idx]))
                row2.append(int(self.rows[partner_idx]))
                targets_all.append(float(target))
                anchor_classes_all.append(anchor_class)

        self.row1 = np.asarray(row1, dtype=np.int64)
        self.row2 = np.asarray(row2, dtype=np.int64)
        self.targets = np.asarray(targets_all, dtype=np.float32)
        self.anchor_classes = np.asarray(anchor_classes_all, dtype=np.int64)

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x1 = np.array(self.fcgr_memmap[int(self.row1[index])], dtype=np.float32, copy=True)
        x2 = np.array(self.fcgr_memmap[int(self.row2[index])], dtype=np.float32, copy=True)
        return {
            "x1": _format_fcgr_tensor(x1),
            "x2": _format_fcgr_tensor(x2),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }


class BalancedSemiHardCandidateDataset(Dataset):
    """
    Dataset bilanciato per il Dynamic Semi-Hard Negative Mining:
    per ogni anchor restituisce 1 positive e `n_negative_candidates` candidati negativi.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        fcgr_memmap: np.ndarray,
        id_to_row: Dict[str, int],
        n_batches: int,
        batch_size: int = 60,
        n_negative_candidates: int = 6,
        seed: int = 42,
    ) -> None:
        self.metadata = metadata.copy().reset_index(drop=True)
        self.fcgr_memmap = fcgr_memmap
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.n_negative_candidates = int(n_negative_candidates)
        self.seed = int(seed)

        self.rows = (
            self.metadata["id"].astype(str).map(id_to_row).to_numpy(dtype=np.int64)
        )
        self.labels = self.metadata["class_id"].to_numpy(dtype=np.int64)
        self.classes = np.array(sorted(np.unique(self.labels)), dtype=np.int64)
        self.n_classes = len(self.classes)

        if self.batch_size % self.n_classes != 0:
            raise ValueError("batch_size must be divisible by number of classes")

        self.samples_per_class_per_batch = self.batch_size // self.n_classes
        self.class_to_indices = {
            int(c): np.where(self.labels == c)[0] for c in self.classes
        }
        self.n_anchors = self.n_batches * self.batch_size
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch) * 100_003)
        anchor_indices, positive_indices, negative_indices, anchor_classes_all = (
            [],
            [],
            [],
            [],
        )

        for _ in range(self.n_batches):
            anchor_classes = np.repeat(
                self.classes, self.samples_per_class_per_batch
            )
            rng.shuffle(anchor_classes)

            for anchor_class in anchor_classes:
                anchor_class = int(anchor_class)
                anchor_idx = int(rng.choice(self.class_to_indices[anchor_class]))

                positive_candidates = self.class_to_indices[anchor_class]
                positive_idx = anchor_idx
                while positive_idx == anchor_idx:
                    positive_idx = int(rng.choice(positive_candidates))

                negative_classes = self.classes[self.classes != anchor_class]
                selected_classes = rng.choice(
                    negative_classes,
                    size=self.n_negative_candidates,
                    replace=(self.n_negative_candidates > len(negative_classes)),
                )
                selected_negatives = [
                    int(rng.choice(self.class_to_indices[int(neg_c)]))
                    for neg_c in selected_classes
                ]

                anchor_indices.append(anchor_idx)
                positive_indices.append(positive_idx)
                negative_indices.append(selected_negatives)
                anchor_classes_all.append(anchor_class)

        self.anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
        self.positive_indices = np.asarray(positive_indices, dtype=np.int64)
        self.negative_indices = np.asarray(negative_indices, dtype=np.int64)
        self.anchor_classes = np.asarray(anchor_classes_all, dtype=np.int64)

    def __len__(self) -> int:
        return self.n_anchors

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        anchor_idx = int(self.anchor_indices[index])
        positive_idx = int(self.positive_indices[index])

        anchor = np.array(
            self.fcgr_memmap[int(self.rows[anchor_idx])], dtype=np.float32, copy=True
        )
        positive = np.array(
            self.fcgr_memmap[int(self.rows[positive_idx])], dtype=np.float32, copy=True
        )

        negatives_list = [
            _format_fcgr_tensor(
                np.array(
                    self.fcgr_memmap[int(self.rows[int(neg_idx)])],
                    dtype=np.float32,
                    copy=True,
                )
            )
            for neg_idx in self.negative_indices[index]
        ]

        return {
            "anchor": _format_fcgr_tensor(anchor),
            "positive": _format_fcgr_tensor(positive),
            "negatives": torch.stack(negatives_list, dim=0),
        }


class SingleFCGRDataset(Dataset):
    """
    Dataset per estrarre gli embedding singoli (reference bank + validation/test ranking).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        fcgr_memmap: np.ndarray,
        id_to_row: Dict[str, int],
    ) -> None:
        self.metadata = metadata.copy().reset_index(drop=True)
        self.fcgr_memmap = fcgr_memmap
        self.rows = (
            self.metadata["id"].astype(str).map(id_to_row).to_numpy(dtype=np.int64)
        )
        self.labels = self.metadata["class_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = np.array(
            self.fcgr_memmap[int(self.rows[index])], dtype=np.float32, copy=True
        )
        return {
            "x": _format_fcgr_tensor(x),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }

