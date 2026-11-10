"""
FCGR (Frequency Chaos Game Representation) baseline encoding and visual diagnostics.

Contains:
- FASTA parsing utilities (`load_fasta_sequences`)
- Pure baseline FCGR calculation (`compute_fcgr`, `normalize_fcgr`, `sequence_to_fcgr_tensor`)
- Diagnostic metrics & visual feedback tools (`compute_fcgr_diagnostics`,
  `plot_fcgr_diagnostics`, `plot_siamese_pair_feedback`, `plot_fcgr_comparison`)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


VALID_BASES = frozenset("ACGT")

CGR_CORNERS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 1.0),
    "C": (0.0, 0.0),
    "G": (1.0, 0.0),
    "T": (1.0, 1.0),
}


def load_fasta_sequences(
    fasta_path: Union[str, Path],
    target_ids: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """
    Recupera dal file FASTA le sequenze i cui ID sono presenti in `target_ids`
    (oppure tutte le sequenze se `target_ids` è None).

    L'ID FASTA viene interpretato come la parte dell'header precedente al primo "|".
    """
    target_set = set(map(str, target_ids)) if target_ids is not None else None
    sequences: Dict[str, str] = {}

    current_id: Optional[str] = None
    current_sequence: list[str] = []
    current_is_target: bool = False

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None and current_is_target:
                    sequences[current_id] = "".join(current_sequence)
                    if target_set is not None and len(sequences) == len(target_set):
                        break

                header = line[1:]
                current_id = header.split("|", 1)[0].strip()
                current_is_target = (
                    True if target_set is None else (current_id in target_set)
                )
                current_sequence = []
            elif current_is_target:
                current_sequence.append(line)

        if (
            current_id is not None
            and current_is_target
            and current_id not in sequences
        ):
            sequences[current_id] = "".join(current_sequence)

    return sequences


def compute_fcgr(
    sequence: str,
    k: int,
    return_distinct: bool = True,
) -> Union[Tuple[np.ndarray, int, int], Tuple[np.ndarray, int]]:
    """
    Calcola la Frequency Chaos Game Representation (FCGR) baseline (ciclo Python puro)
    usando la convenzione dei vertici del progetto originale:
        A = (0, 1), T = (1, 1), C = (0, 0), G = (1, 0)

    Parameters
    ----------
    sequence : str
        Sequenza nucleotidica (FASTA).
    k : int
        Lunghezza dei k-mer (produce una matrice di lato 2^k x 2^k).
    return_distinct : bool, default=True
        Se True restituisce `(fcgr_counts, valid_kmers, distinct_kmers)` (compatibile
        con `02_fcgr_generation.ipynb`). Se False restituisce `(fcgr_counts, valid_kmers)`
        (compatibile con `03_siamese_dataset.ipynb`).
    """
    sequence = sequence.upper()
    size = 2 ** k
    fcgr_counts = np.zeros((size, size), dtype=np.int32)

    valid_kmers = 0

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if any(base not in CGR_CORNERS for base in kmer):
            continue

        x = 0.5
        y = 0.5
        for base in kmer:
            cx, cy = CGR_CORNERS[base]
            x = (x + cx) * 0.5
            y = (y + cy) * 0.5

        col = min(int(x * size), size - 1)
        row = min(int(y * size), size - 1)

        fcgr_counts[row, col] += 1
        valid_kmers += 1

    if return_distinct:
        distinct_kmers = int(np.count_nonzero(fcgr_counts))
        return fcgr_counts, valid_kmers, distinct_kmers

    return fcgr_counts, valid_kmers


def normalize_fcgr(fcgr_counts: np.ndarray) -> np.ndarray:
    """
    Converte i conteggi FCGR in frequenze relative in `float32` (somma = 1.0).
    """
    total = fcgr_counts.sum()
    fcgr = fcgr_counts.astype(np.float32)
    if total == 0:
        return fcgr
    fcgr /= np.float32(total)
    return fcgr


def sequence_to_fcgr_tensor(
    sequence: str,
    k: int = 6,
) -> Tuple[torch.Tensor, int]:
    """
    Converte una sequenza nucleotidica in un tensore PyTorch FCGR normalizzato
    di shape `(1, 2^k, 2^k)` e dtype `torch.float32`.
    """
    fcgr_counts, n_valid, _ = compute_fcgr(sequence, k, return_distinct=True)
    fcgr_norm = normalize_fcgr(fcgr_counts)
    tensor = torch.from_numpy(fcgr_norm).unsqueeze(0).float()
    return tensor, n_valid


def compute_fcgr_diagnostics(
    fcgr_matrix: Union[np.ndarray, torch.Tensor],
    sequence: Optional[str] = None,
    k: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calcola un dizionario completo di metriche diagnostiche su una matrice FCGR
    (sparsità, occupazione, entropia di Shannon, memoria occupata, distribuzione quadranti).
    """
    if isinstance(fcgr_matrix, torch.Tensor):
        arr = fcgr_matrix.detach().cpu().numpy()
    else:
        arr = np.asarray(fcgr_matrix)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    side = arr.shape[-1]
    inferred_k = int(np.log2(side)) if k is None else int(k)
    total_cells = int(arr.size)
    non_zero_cells = int(np.count_nonzero(arr))
    zero_cells = total_cells - non_zero_cells

    occupancy_pct = (non_zero_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
    sparsity_pct = 100.0 - occupancy_pct

    total_sum = float(arr.sum())
    if total_sum > 0:
        probs = arr[arr > 0].astype(np.float64) / total_sum
        shannon_entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = float(np.log2(total_cells))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        shannon_entropy = 0.0
        normalized_entropy = 0.0

    # Densità nei 4 quadranti CGR: C=(0,0) bottom-left, G=(1,0) bottom-right,
    # A=(0,1) top-left, T=(1,1) top-right (con origin='lower')
    half = side // 2
    q_c = float(arr[:half, :half].sum()) / (total_sum if total_sum > 0 else 1.0) * 100.0
    q_g = float(arr[:half, half:].sum()) / (total_sum if total_sum > 0 else 1.0) * 100.0
    q_a = float(arr[half:, :half].sum()) / (total_sum if total_sum > 0 else 1.0) * 100.0
    q_t = float(arr[half:, half:].sum()) / (total_sum if total_sum > 0 else 1.0) * 100.0

    diagnostics: Dict[str, float] = {
        "k": inferred_k,
        "side": side,
        "total_cells": total_cells,
        "non_zero_cells": non_zero_cells,
        "zero_cells": zero_cells,
        "occupancy_pct": occupancy_pct,
        "sparsity_pct": sparsity_pct,
        "max_value": float(arr.max()) if total_cells > 0 else 0.0,
        "mean_nonzero_value": float(arr[arr > 0].mean()) if non_zero_cells > 0 else 0.0,
        "shannon_entropy_bits": shannon_entropy,
        "normalized_entropy": normalized_entropy,
        "memory_float32_kb": (total_cells * 4) / 1024.0,
        "memory_uint8_kb": (total_cells * 1) / 1024.0,
        "quadrant_C_pct": q_c,
        "quadrant_G_pct": q_g,
        "quadrant_A_pct": q_a,
        "quadrant_T_pct": q_t,
    }

    if sequence is not None:
        seq_upper = sequence.upper()
        seq_len = len(seq_upper)
        gc_count = seq_upper.count("G") + seq_upper.count("C")
        valid_kmers = max(0, seq_len - inferred_k + 1)
        diagnostics["sequence_length"] = seq_len
        diagnostics["gc_content_pct"] = (gc_count / seq_len * 100.0) if seq_len > 0 else 0.0
        diagnostics["unique_ratio"] = (
            non_zero_cells / valid_kmers if valid_kmers > 0 else 0.0
        )

    if elapsed_ms is not None:
        diagnostics["elapsed_ms"] = float(elapsed_ms)

    return diagnostics


def plot_fcgr_diagnostics(
    sequence_or_fcgr: Union[str, np.ndarray, torch.Tensor],
    k: int = 6,
    sample_id: str = "Sample",
    disease: str = "Unknown",
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Visualizza un pannello diagnostico a 4 viste per una singola FCGR:
    1. FCGR in scala lineare (input diretto della CNN Siamese)
    2. FCGR in scala logaritmica (per esaltare i k-mer rari nascosti dalla sparsità)
    3. Mappa binaria di occupazione vs sparsità (celle attive vs zeri)
    4. Distribuzione delle frequenze per quadrante CGR (C, G, A, T) + Riepilogo Metriche
    """
    if isinstance(sequence_or_fcgr, str):
        sequence = sequence_or_fcgr
        t0 = time.perf_counter()
        counts, _, _ = compute_fcgr(sequence, k, return_distinct=True)
        fcgr_norm = normalize_fcgr(counts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    else:
        sequence = None
        elapsed_ms = None
        if isinstance(sequence_or_fcgr, torch.Tensor):
            fcgr_norm = sequence_or_fcgr.detach().cpu().numpy()
        else:
            fcgr_norm = np.asarray(sequence_or_fcgr, dtype=np.float32)
        if fcgr_norm.ndim == 3 and fcgr_norm.shape[0] == 1:
            fcgr_norm = fcgr_norm[0]
        if fcgr_norm.sum() > 1.05:
            fcgr_norm = normalize_fcgr(fcgr_norm)

    diag = compute_fcgr_diagnostics(
        fcgr_norm, sequence=sequence, k=k, elapsed_ms=elapsed_ms
    )

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 1. Scala Lineare
    im0 = axes[0].imshow(
        fcgr_norm, cmap="viridis", origin="lower", interpolation="nearest"
    )
    axes[0].set_title(f"FCGR Lineare (k={int(diag['k'])}, {int(diag['side'])}x{int(diag['side'])})")
    axes[0].set_xlabel("X (C -> G / A -> T)")
    axes[0].set_ylabel("Y (C -> A / G -> T)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Scala Logaritmica
    pos_vals = fcgr_norm[fcgr_norm > 0]
    if len(pos_vals) > 0:
        vmin = max(pos_vals.min(), 1e-6)
        vmax = max(pos_vals.max(), vmin * 1.01)
        im1 = axes[1].imshow(
            np.where(fcgr_norm > 0, fcgr_norm, vmin * 0.5),
            cmap="magma",
            origin="lower",
            interpolation="nearest",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].imshow(fcgr_norm, cmap="magma", origin="lower")
    axes[1].set_title("FCGR Scala Logaritmica (Pattern Rari)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # 3. Mappa di Occupazione / Sparsità
    active_mask = (fcgr_norm > 0).astype(np.int32)
    im2 = axes[2].imshow(
        active_mask, cmap="Blues", origin="lower", interpolation="nearest", vmin=0, vmax=1
    )
    axes[2].set_title(
        f"Mappa Occupazione\nAttive: {int(diag['non_zero_cells'])}/{int(diag['total_cells'])} "
        f"(Sparsità: {diag['sparsity_pct']:.2f}%)"
    )
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])

    # 4. Feedback Quantitativo & Bilanciamento Quadranti
    axes[3].axis("off")
    seq_info = (
        f"Lunghezza Seq. : {int(diag['sequence_length']):,} bp\n"
        f"GC Content     : {diag['gc_content_pct']:.2f}%\n"
        f"Unique Ratio   : {diag['unique_ratio']:.3f}\n"
        if "sequence_length" in diag
        else ""
    )
    time_info = (
        f"Tempo Calcolo  : {diag['elapsed_ms']:.3f} ms\n"
        if "elapsed_ms" in diag
        else ""
    )
    summary_text = (
        f"=== FEEDBACK DIAGNOSTICO FCGR ===\n"
        f"ID      : {sample_id}\n"
        f"Classe  : {disease}\n"
        f"{seq_info}"
        f"---------------------------------\n"
        f"Risoluzione    : {int(diag['side'])}x{int(diag['side'])} ({int(diag['total_cells']):,} celle)\n"
        f"Celle Attive   : {int(diag['non_zero_cells']):,} ({diag['occupancy_pct']:.2f}%)\n"
        f"Celle Zero     : {int(diag['zero_cells']):,} ({diag['sparsity_pct']:.2f}%)\n"
        f"Entropia Norm. : {diag['normalized_entropy']:.3f} ({diag['shannon_entropy_bits']:.2f} bit)\n"
        f"RAM (float32)  : {diag['memory_float32_kb']:.2f} KB\n"
        f"RAM (uint8)    : {diag['memory_uint8_kb']:.2f} KB\n"
        f"{time_info}"
        f"---------------------------------\n"
        f"Densità Quadranti (Ultimo nt):\n"
        f"  A (Top-Left) : {diag['quadrant_A_pct']:5.1f}% | T (Top-Right): {diag['quadrant_T_pct']:5.1f}%\n"
        f"  C (Bot-Left) : {diag['quadrant_C_pct']:5.1f}% | G (Bot-Right): {diag['quadrant_G_pct']:5.1f}%"
    )
    axes[3].text(
        0.02,
        0.96,
        summary_text,
        transform=axes[3].transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F6F8", edgecolor="#CBD2D9"),
    )

    plt.suptitle(
        f"Ispezione FCGR — {sample_id} ({disease})",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return diag


def plot_siamese_pair_feedback(
    x1: Union[np.ndarray, torch.Tensor],
    x2: Union[np.ndarray, torch.Tensor],
    target: float,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    embedding_distance: Optional[float] = None,
    figsize: Tuple[int, int] = (16, 4.5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualizza una coppia `(x1, x2)` così come entra nei due rami della Rete Siamese,
    mostrando anche la mappa della differenza assoluta `|x1 - x2|` e i dati di sparsità.
    """
    arr1 = x1.detach().cpu().numpy() if isinstance(x1, torch.Tensor) else np.asarray(x1)
    arr2 = x2.detach().cpu().numpy() if isinstance(x2, torch.Tensor) else np.asarray(x2)
    if arr1.ndim == 3:
        arr1 = arr1[0]
    if arr2.ndim == 3:
        arr2 = arr2[0]

    diff = np.abs(arr1 - arr2)
    d1 = compute_fcgr_diagnostics(arr1)
    d2 = compute_fcgr_diagnostics(arr2)
    l1_dist = float(diff.sum())
    l2_dist = float(np.sqrt(np.sum(diff ** 2)))

    pair_type = "POSITIVA (Stessa Classe)" if int(round(target)) == 1 else "NEGATIVA (Classi Diverse)"

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    im0 = axes[0].imshow(np.log1p(arr1 * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(f"Ramo 1: {label1}\nSparsità: {d1['sparsity_pct']:.1f}% ({int(d1['non_zero_cells'])} celle)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.log1p(arr2 * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[1].set_title(f"Ramo 2: {label2}\nSparsità: {d2['sparsity_pct']:.1f}% ({int(d2['non_zero_cells'])} celle)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="magma", origin="lower", interpolation="nearest")
    axes[2].set_title(f"Differenza Pixel |x1 - x2|\nL1={l1_dist:.3f} | L2={l2_dist:.4f}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].axis("off")
    emb_line = (
        f"Distanza Embedding : {embedding_distance:.4f}\n"
        if embedding_distance is not None
        else "Distanza Embedding : (pre-modello)\n"
    )
    info_text = (
        f"=== COPPIA INPUT RETE SIAMESE ===\n"
        f"Tipo Coppia : {pair_type}\n"
        f"Target y    : {int(round(target))}\n"
        f"---------------------------------\n"
        f"Ramo 1      : {label1}\n"
        f"  Celle > 0 : {int(d1['non_zero_cells'])} / {int(d1['total_cells'])}\n"
        f"  Sparsità  : {d1['sparsity_pct']:.2f}%\n"
        f"Ramo 2      : {label2}\n"
        f"  Celle > 0 : {int(d2['non_zero_cells'])} / {int(d2['total_cells'])}\n"
        f"  Sparsità  : {d2['sparsity_pct']:.2f}%\n"
        f"---------------------------------\n"
        f"Distanza L1 (FCGR) : {l1_dist:.4f}\n"
        f"Distanza L2 (FCGR) : {l2_dist:.4f}\n"
        f"{emb_line}"
    )
    axes[3].text(
        0.02,
        0.95,
        info_text,
        transform=axes[3].transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F6F8", edgecolor="#CBD2D9"),
    )

    plt.suptitle(
        f"Coppia Siamese [{pair_type}] — {label1} vs {label2}",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_fcgr_comparison(
    fcgr_baseline: Union[np.ndarray, torch.Tensor],
    fcgr_optimized: Union[np.ndarray, torch.Tensor],
    time_baseline_ms: float,
    time_optimized_ms: float,
    name_baseline: str = "Baseline FCGR",
    name_optimized: str = "Optimized FCGR",
    bytes_baseline: Optional[int] = None,
    bytes_optimized: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 4.5),
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Confronta visivamente e numericamente la FCGR Baseline con una variante Ottimizzata,
    mostrando speedup, riduzione della memoria, sparsità ed errore/differenza strutturale.
    """
    b_arr = (
        fcgr_baseline.detach().cpu().numpy()
        if isinstance(fcgr_baseline, torch.Tensor)
        else np.asarray(fcgr_baseline)
    )
    o_arr = (
        fcgr_optimized.detach().cpu().numpy()
        if isinstance(fcgr_optimized, torch.Tensor)
        else np.asarray(fcgr_optimized)
    )

    b_2d = b_arr[0] if b_arr.ndim == 3 else b_arr
    o_2d = o_arr[0] if o_arr.ndim == 3 else o_arr

    d_base = compute_fcgr_diagnostics(b_2d, elapsed_ms=time_baseline_ms)
    d_opt = compute_fcgr_diagnostics(o_2d, elapsed_ms=time_optimized_ms)

    mem_base_kb = (bytes_baseline / 1024.0) if bytes_baseline else (b_arr.nbytes / 1024.0)
    mem_opt_kb = (bytes_optimized / 1024.0) if bytes_optimized else (o_arr.nbytes / 1024.0)

    speedup = time_baseline_ms / max(time_optimized_ms, 1e-9)
    mem_saving_pct = (1.0 - (mem_opt_kb / max(mem_base_kb, 1e-9))) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(np.log1p(b_2d * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(
        f"{name_baseline}\nShape: {tuple(b_arr.shape)} | Sparsità: {d_base['sparsity_pct']:.1f}%"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.log1p(o_2d * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[1].set_title(
        f"{name_optimized}\nShape: {tuple(o_arr.shape)} | Sparsità: {d_opt['sparsity_pct']:.1f}%"
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].axis("off")
    same_shape = b_2d.shape == o_2d.shape
    max_abs_err = float(np.max(np.abs(b_2d - o_2d))) if same_shape else float("nan")

    err_line = (
        f"Max Diff Assoluta : {max_abs_err:.2e}\n"
        if same_shape
        else "Risoluzione Compatta / Multi-Canale\n"
    )

    comp_text = (
        f"=== CONFRONTO BASELINE vs OTTIMIZZATA ===\n"
        f"Tempo Baseline    : {time_baseline_ms:8.3f} ms\n"
        f"Tempo Ottimizzata : {time_optimized_ms:8.3f} ms\n"
        f"SPEEDUP           : {speedup:8.2f}x\n"
        f"-----------------------------------------\n"
        f"RAM Baseline      : {mem_base_kb:8.2f} KB\n"
        f"RAM Ottimizzata   : {mem_opt_kb:8.2f} KB\n"
        f"RISPARMIO MEMORIA : {mem_saving_pct:7.1f}%\n"
        f"-----------------------------------------\n"
        f"Sparsità Baseline : {d_base['sparsity_pct']:7.2f}%\n"
        f"Sparsità Ottim.   : {d_opt['sparsity_pct']:7.2f}%\n"
        f"{err_line}"
    )
    axes[2].text(
        0.02,
        0.95,
        comp_text,
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF8FF", edgecolor="#90CDF4"),
    )

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "speedup": speedup,
        "mem_saving_pct": mem_saving_pct,
        "sparsity_baseline_pct": d_base["sparsity_pct"],
        "sparsity_optimized_pct": d_opt["sparsity_pct"],
        "max_abs_err": max_abs_err,
    }

