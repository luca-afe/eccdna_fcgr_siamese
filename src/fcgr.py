"""
FCGR (Frequency Chaos Game Representation) baseline & optimized encodings,
multi-scale / reverse-complement representations, quantization, and visual diagnostics.

Contains:
- FASTA parsing utilities (`load_fasta_sequences`, `reverse_complement`)
- Pure baseline FCGR calculation (`compute_fcgr`, `normalize_fcgr`, `sequence_to_fcgr_tensor`)
- Optimized Fase 2 FCGR algorithms:
    * `compute_fcgr_vectorized` (Bitwise NumPy vectorization, 25x-35x speedup, optional Reverse-Complement)
    * `normalize_fcgr_log` (Length-robust log1p normalization to [0, 1])
    * `compute_multiscale_fcgr` (Multi-Scale fractal-aligned stack, e.g. k=(4, 5, 6) -> (3, 64, 64))
    * `quantize_fcgr_uint8` / `dequantize_fcgr_uint8` (4x memory compression vs float32)
    * `sequence_to_optimized_fcgr_tensor`
- Diagnostic metrics & visual feedback tools:
    * `compute_fcgr_diagnostics`
    * `plot_fcgr_diagnostics`
    * `plot_multik_progression`
    * `plot_multiscale_fcgr_diagnostics`
    * `plot_siamese_pair_feedback`
    * `plot_fcgr_comparison`
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - allows pure NumPy benchmarking when torch is absent locally
    torch = None  # type: ignore


VALID_BASES = frozenset("ACGT")

CGR_CORNERS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 1.0),
    "C": (0.0, 0.0),
    "G": (1.0, 0.0),
    "T": (1.0, 1.0),
}

_RC_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_K_LUTS: Dict[int, np.ndarray] = {}


def reverse_complement(sequence: str) -> str:
    """
    Restituisce il filamento complementare inverso (5' -> 3') di una sequenza DNA.
    """
    return sequence.translate(_RC_TABLE)[::-1]


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


# =============================================================================
# 1. BASELINE FCGR (Ciclo Python Originale)
# =============================================================================


def compute_fcgr(
    sequence: str,
    k: int,
    return_distinct: bool = True,
) -> Union[Tuple[np.ndarray, int, int], Tuple[np.ndarray, int]]:
    """
    Calcola la Frequency Chaos Game Representation (FCGR) baseline (ciclo Python puro)
    usando la convenzione dei vertici del progetto originale:
        A = (0, 1), T = (1, 1), C = (0, 0), G = (1, 0)
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
) -> Tuple["torch.Tensor", int]:
    """
    Converte una sequenza nucleotidica in un tensore PyTorch FCGR normalizzato
    di shape `(1, 2^k, 2^k)` e dtype `torch.float32`.
    """
    if torch is None:
        raise ImportError("PyTorch è richiesto per sequence_to_fcgr_tensor.")
    fcgr_counts, n_valid, _ = compute_fcgr(sequence, k, return_distinct=True)
    fcgr_norm = normalize_fcgr(fcgr_counts)
    tensor = torch.from_numpy(fcgr_norm).unsqueeze(0).float()
    return tensor, n_valid


# =============================================================================
# 2. FASE 2 — ALGORITMI FCGR OTTIMIZZATI (Bitwise, RC, Log-Norm, Multi-Scale, Quantizzazione)
# =============================================================================


def _get_k_lut(k: int) -> np.ndarray:
    """
    Costruisce e memorizza in cache la Lookup Table (256 voci ASCII) per l'ordine `k`.
    Ogni base valida codifica `(cy << k) | cx`, mentre i caratteri non-ACGT attivano
    il bit di invalidazione `1 << (2 * k)`.
    """
    if k not in _K_LUTS:
        invalid_bit = 1 << (2 * k)
        lut = np.full(256, invalid_bit, dtype=np.int32)
        for ch, (cx, cy) in CGR_CORNERS.items():
            val = (int(cy) << k) | int(cx)
            lut[ord(ch)] = val
            lut[ord(ch.lower())] = val
        _K_LUTS[k] = lut
    return _K_LUTS[k]


def _count_flat_kmers_bitwise(sequence: str, k: int) -> Tuple[np.ndarray, int]:
    """
    Calcola in modo vettorizzato gli indici lineari `(row << k) | col` di tutti i k-mer validi.
    """
    n = len(sequence)
    limit = 1 << (2 * k)
    if n < k:
        return np.empty(0, dtype=np.int32), 0

    lut = _get_k_lut(k)
    ascii_arr = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
    encoded = lut[ascii_arr]

    n_windows = n - k + 1
    flat_idx = encoded[:n_windows].copy()
    for t in range(1, k):
        flat_idx |= encoded[t : t + n_windows] << t

    if np.any(flat_idx >= limit):
        flat_idx = flat_idx[flat_idx < limit]

    return flat_idx, int(flat_idx.size)


def compute_fcgr_vectorized(
    sequence: str,
    k: int,
    return_distinct: bool = True,
    include_reverse_complement: bool = False,
) -> Union[Tuple[np.ndarray, int, int], Tuple[np.ndarray, int]]:
    """
    Calcola la FCGR in modo interamente vettorizzato tramite bit-shift NumPy.
    Produce esattamente gli stessi conteggi di `compute_fcgr` (errore 0.0), ma ~25x-35x più velocemente.

    Parameters
    ----------
    sequence : str
        Sequenza nucleotidica.
    k : int
        Ordine del k-mer (griglia 2^k x 2^k).
    return_distinct : bool, default=True
        Se True restituisce `(fcgr_counts, valid_kmers, distinct_kmers)`.
    include_reverse_complement : bool, default=False
        Se True somma i conteggi del filamento Forward e del Reverse-Complement,
        rendendo la rappresentazione invariante al filamento (Canonical Strand-Invariant FCGR).
    """
    size = 1 << k
    limit = 1 << (2 * k)

    flat_fwd, valid_fwd = _count_flat_kmers_bitwise(sequence, k)
    if include_reverse_complement:
        flat_rc, valid_rc = _count_flat_kmers_bitwise(reverse_complement(sequence), k)
        flat_all = np.concatenate([flat_fwd, flat_rc]) if (valid_fwd + valid_rc) > 0 else flat_fwd
        valid_kmers = valid_fwd + valid_rc
    else:
        flat_all = flat_fwd
        valid_kmers = valid_fwd

    if valid_kmers == 0:
        zeros = np.zeros((size, size), dtype=np.int32)
        return (zeros, 0, 0) if return_distinct else (zeros, 0)

    if k <= 7:
        counts_1d = np.bincount(flat_all, minlength=limit)
        fcgr_counts = counts_1d.astype(np.int32).reshape((size, size))
        distinct_kmers = int(np.count_nonzero(counts_1d))
    else:
        fcgr_counts = np.zeros((size, size), dtype=np.int32)
        np.add.at(fcgr_counts.ravel(), flat_all, 1)
        distinct_kmers = int(np.count_nonzero(fcgr_counts))

    if return_distinct:
        return fcgr_counts, valid_kmers, distinct_kmers
    return fcgr_counts, valid_kmers


def normalize_fcgr_log(
    fcgr_counts: np.ndarray,
    scale_factor: float = 1000.0,
) -> np.ndarray:
    """
    Normalizzazione sub-lineare Log1p + Max-Scaling in `[0.0, 1.0]` per mitigare il bias
    di lunghezza della sequenza (evita che sequenze corte abbiano pochi picchi giganti
    e sequenze lunghe abbiano valori schiacciati verso zero):
        1. Frequenza per `scale_factor` k-mer: p = (counts / sum(counts)) * scale_factor
        2. Compressione logaritmica: L = log(1 + p)
        3. Scaling sul massimo in [0, 1]: L / max(L)
    """
    total = float(fcgr_counts.sum())
    if total <= 0.0:
        return np.zeros_like(fcgr_counts, dtype=np.float32)

    rel_freq = (fcgr_counts.astype(np.float32) / np.float32(total)) * np.float32(scale_factor)
    log_fcgr = np.log1p(rel_freq)
    max_val = float(log_fcgr.max())
    if max_val > 0.0:
        log_fcgr /= np.float32(max_val)
    return log_fcgr.astype(np.float32)


def normalize_fcgr_log_prob(fcgr_counts: np.ndarray) -> np.ndarray:
    """
    Normalizzazione sub-lineare Log1p a somma unitaria (`sum = 1.0`):
        L = log(1 + counts)
        P_log = L / sum(L)
    Elimina il bias del singolo picco massimo (`/ max_val`) mantenendo esattamente
    il regime di varianza (`~5e-7`) atteso da `GroupNorm` in `SiameseV3`.
    """
    log_c = np.log1p(fcgr_counts.astype(np.float32))
    total = float(log_c.sum())
    if total <= 0.0:
        return np.zeros_like(log_c, dtype=np.float32)
    return (log_c / np.float32(total)).astype(np.float32)


_GRID_BASE_COUNTS: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_grid_base_counts(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precalcola per ogni cella `(r, c)` della griglia `2^k x 2^k` il numero esatto di
    basi `(nA, nC, nG, nT)` che compongono il k-mer corrispondente.
    """
    if k not in _GRID_BASE_COUNTS:
        side = 1 << k
        rows, cols = np.indices((side, side), dtype=np.int32)
        n_a = np.zeros((side, side), dtype=np.int32)
        n_c = np.zeros((side, side), dtype=np.int32)
        n_g = np.zeros((side, side), dtype=np.int32)
        n_t = np.zeros((side, side), dtype=np.int32)
        for t in range(k):
            cx = (cols >> t) & 1
            cy = (rows >> t) & 1
            n_c += (cx == 0) & (cy == 0)
            n_g += (cx == 1) & (cy == 0)
            n_a += (cx == 0) & (cy == 1)
            n_t += (cx == 1) & (cy == 1)
        _GRID_BASE_COUNTS[k] = (n_a, n_c, n_g, n_t)
    return _GRID_BASE_COUNTS[k]


def compute_oe_enrichment_fcgr(
    sequence: str,
    k: int = 6,
    counts_fwd: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calcola la FCGR di Arricchimento Sovra-Casuale (Observed / Expected, O/E)
    rispetto alla composizione nucleotidica (A, C, G, T) della sequenza stessa:
        P_exp(r, c) = (p_A^nA) * (p_C^nC) * (p_G^nG) * (p_T^nT)
        E(r, c) = log(1 + P_obs(r, c) / (P_exp(r, c) + 1e-5))
    Normalizzata a somma unitaria (`sum = 1.0`).
    Rimuove il rumore di fondo dovuto al semplice contenuto GC/AT ed esalta
    solo i motivi biologicamente sovra-rappresentati (es. isole CpG, micro-omologie).
    """
    if counts_fwd is None:
        counts_fwd, _, _ = compute_fcgr_vectorized(
            sequence, k=k, return_distinct=True, include_reverse_complement=False
        )
    p_obs = normalize_fcgr(counts_fwd)
    if float(p_obs.sum()) <= 0.0:
        return p_obs

    seq_u = sequence.upper()
    cnt_a = seq_u.count("A") + 1.0
    cnt_c = seq_u.count("C") + 1.0
    cnt_g = seq_u.count("G") + 1.0
    cnt_t = seq_u.count("T") + 1.0
    tot_bases = cnt_a + cnt_c + cnt_g + cnt_t

    pa = np.float32(cnt_a / tot_bases)
    pc = np.float32(cnt_c / tot_bases)
    pg = np.float32(cnt_g / tot_bases)
    pt = np.float32(cnt_t / tot_bases)

    n_a, n_c, n_g, n_t = _get_grid_base_counts(k)
    exps = np.arange(k + 1, dtype=np.float32)
    p_exp = (
        (pa ** exps)[n_a]
        * (pc ** exps)[n_c]
        * (pg ** exps)[n_g]
        * (pt ** exps)[n_t]
    )

    oe_ratio = p_obs / (p_exp.astype(np.float32) + np.float32(1e-5))
    enrichment = np.log1p(oe_ratio)
    s = float(enrichment.sum())
    if s > 0.0:
        enrichment /= np.float32(s)
    return enrichment.astype(np.float32)


def compute_opt3_enriched_fcgr(
    sequence: str,
    target_k: int = 6,
) -> Tuple[np.ndarray, int]:
    """
    Costruisce la rappresentazione `Opt 3` a 3 canali fisicamente bilanciati (`sum = 1.0`
    su ciascun canale) di shape `(3, 2^target_k, 2^target_k)`:
      - Canale 0: `k=6 Forward Log-Prob` (`sum = 1.0`, dettaglio fine 6-mer, filamento originale,
                  senza distorsione `/ max_val`).
      - Canale 1: `k=6 Observed/Expected Motif Enrichment` (`sum = 1.0`, esalta i motivi
                  biologici sovra-rappresentati ripulendoli dal semplice contenuto GC/AT).
      - Canale 2: `Multi-Scale Fractal Context (k=4 + k=5 + RC)` (`sum = 1.0`, abbatte la
                  sparsità spaziale da ~88% a ~13-18% con energia perfettamente bilanciata).
    """
    c6_fwd, valid_k6, _ = compute_fcgr_vectorized(
        sequence, k=target_k, return_distinct=True, include_reverse_complement=False
    )
    ch0_log_prob = normalize_fcgr_log_prob(c6_fwd)
    ch1_oe = compute_oe_enrichment_fcgr(sequence, k=target_k, counts_fwd=c6_fwd)

    k_coarse = max(1, target_k - 2)
    k_mid = max(1, target_k - 1)
    c4_rc, _, _ = compute_fcgr_vectorized(sequence, k=k_coarse, return_distinct=True, include_reverse_complement=True)
    c5_rc, _, _ = compute_fcgr_vectorized(sequence, k=k_mid, return_distinct=True, include_reverse_complement=True)

    p4 = normalize_fcgr_log_prob(c4_rc)
    p5 = normalize_fcgr_log_prob(c5_rc)

    s4 = 1 << (target_k - k_coarse)
    s5 = 1 << (target_k - k_mid)
    u4 = np.repeat(np.repeat(p4, s4, axis=0), s4, axis=1) / np.float32(s4 * s4)
    u5 = np.repeat(np.repeat(p5, s5, axis=0), s5, axis=1) / np.float32(s5 * s5)

    ch2_ctx = np.float32(0.5) * u4 + np.float32(0.5) * u5
    ctx_sum = float(ch2_ctx.sum())
    if ctx_sum > 0.0:
        ch2_ctx /= np.float32(ctx_sum)

    stack = np.stack([ch0_log_prob, ch1_oe, ch2_ctx.astype(np.float32)], axis=0)
    return stack, valid_k6


def quantize_fcgr_uint8(fcgr_float: np.ndarray) -> np.ndarray:
    """
    Quantizza una FCGR 2D `(H, W)` o 3D `(C, H, W)` in interi a 8 bit `uint8` (`0..255`),
    scalando ciascun canale sul proprio massimo per sfruttare tutti i 256 livelli.
    Quando viene riletta con `dequantize_fcgr_uint8_prob` (normalizzazione a `sum = 1.0`),
    il fattore `255 / max_val` si cancella algebricamente!
    """
    arr = np.asarray(fcgr_float, dtype=np.float32)
    if arr.ndim == 3:
        max_val = arr.max(axis=(-2, -1), keepdims=True)
        max_val = np.where(max_val > 0.0, max_val, 1.0)
        scaled = np.clip((arr / max_val) * 255.0, 0.0, 255.0)
        return np.rint(scaled).astype(np.uint8)

    max_val_scalar = float(arr.max())
    if max_val_scalar <= 0.0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr / np.float32(max_val_scalar)) * 255.0, 0.0, 255.0)
    return np.rint(scaled).astype(np.uint8)


def dequantize_fcgr_uint8(fcgr_u8: np.ndarray) -> np.ndarray:
    """
    Riconverte una FCGR quantizzata `uint8` in `float32` nel range `[0.0, 1.0]`.
    """
    return np.asarray(fcgr_u8, dtype=np.float32) * np.float32(1.0 / 255.0)


def dequantize_fcgr_uint8_prob(fcgr_u8: np.ndarray) -> np.ndarray:
    """
    Dequantizza una FCGR `uint8` `(H, W)` o `(C, H, W)` normalizzando ciascun canale
    a somma unitaria (`sum = 1.0`). Cancella algebricamente la divisione per `max_val`
    e ripristina il regime di varianza esatto (`~5e-7`) per `SiameseV3`.
    """
    arr = np.asarray(fcgr_u8, dtype=np.float32)
    if arr.ndim == 2:
        s = float(arr.sum())
        return (arr / np.float32(s)) if s > 0.0 else arr
    if arr.ndim == 3:
        s = arr.sum(axis=(-2, -1), keepdims=True)
        s = np.where(s > 0.0, s, 1.0)
        return (arr / s).astype(np.float32)
    return arr


def compute_multiscale_fcgr(
    sequence: str,
    ks: Sequence[int] = (4, 5, 6),
    target_k: int = 6,
    include_reverse_complement: bool = True,
    log_normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Costruisce una rappresentazione FCGR Multi-Scala e Multi-Canale di shape
    `(len(ks), 2^target_k, 2^target_k)` sfruttando la proprietà di autosimilarità frattale CGR:
    ogni cella di ordine `k_c < target_k` è esattamente il quadrante padre di un blocco
    `2^(target_k - k_c) x 2^(target_k - k_c)` nella griglia `2^target_k x 2^target_k`.

    Per default con `ks=(4, 5, 6)` e `target_k=6` produce un tensore `(3, 64, 64)`:
      - Canale 0 (k=4, 16x16 -> 64x64): contesto macro-composizionale denso (~15-25% sparsità)
      - Canale 1 (k=5, 32x32 -> 64x64): contesto intermedio (meso-scala)
      - Canale 2 (k=6, 64x64)         : alta risoluzione spaziale dei 6-mer

    Returns
    -------
    stack : np.ndarray of shape `(C, 2^target_k, 2^target_k)`, dtype `float32`
    valid_target_kmers : int
    """
    target_side = 1 << target_k
    channels = []
    valid_target_kmers = 0

    for k_val in ks:
        if k_val > target_k:
            raise ValueError(f"Ogni k in ks ({ks}) deve essere <= target_k ({target_k}).")

        counts, n_valid, _ = compute_fcgr_vectorized(
            sequence,
            k=k_val,
            return_distinct=True,
            include_reverse_complement=include_reverse_complement,
        )
        if k_val == target_k:
            valid_target_kmers = n_valid

        norm_mat = normalize_fcgr_log(counts) if log_normalize else normalize_fcgr(counts)

        scale = 1 << (target_k - k_val)
        if scale > 1:
            upsampled = np.repeat(np.repeat(norm_mat, scale, axis=0), scale, axis=1)
        else:
            upsampled = norm_mat

        assert upsampled.shape == (target_side, target_side)
        channels.append(upsampled.astype(np.float32))

    stack = np.stack(channels, axis=0)
    return stack, valid_target_kmers


def sequence_to_optimized_fcgr_tensor(
    sequence: str,
    k: int = 6,
    mode: str = "multiscale",
    include_reverse_complement: bool = True,
) -> Tuple["torch.Tensor", int]:
    """
    Converte una sequenza in un tensore PyTorch `(C, 2^k, 2^k)` usando le varianti della Fase 2:
      - `mode="vectorized"`: baseline k=6 calcolata bitwise (`(1, 64, 64)`)
      - `mode="rc_log"`: k=6 bitwise + Reverse-Complement + Log1p scaling (`(1, 64, 64)`)
      - `mode="multiscale"`: stack multi-scala k=(k-2, k-1, k) + RC + Log1p (`(3, 64, 64)`)
    """
    if torch is None:
        raise ImportError("PyTorch è richiesto per sequence_to_optimized_fcgr_tensor.")

    if mode == "vectorized":
        counts, n_valid, _ = compute_fcgr_vectorized(sequence, k, return_distinct=True, include_reverse_complement=False)
        arr = normalize_fcgr(counts)[None, :, :]
    elif mode == "rc_log":
        counts, n_valid, _ = compute_fcgr_vectorized(
            sequence, k, return_distinct=True, include_reverse_complement=include_reverse_complement
        )
        arr = normalize_fcgr_log(counts)[None, :, :]
    elif mode == "multiscale":
        ks = (max(1, k - 2), max(1, k - 1), k)
        arr, n_valid = compute_multiscale_fcgr(
            sequence,
            ks=ks,
            target_k=k,
            include_reverse_complement=include_reverse_complement,
            log_normalize=True,
        )
    else:
        raise ValueError(f"Modalità FCGR non riconosciuta: {mode}")

    return torch.from_numpy(arr).float(), n_valid


# =============================================================================
# 3. DIAGNOSTICA E VISUALIZZAZIONE (Singola, Multi-k, Multi-Scala, Coppie Siamesi)
# =============================================================================


def compute_fcgr_diagnostics(
    fcgr_matrix: Union[np.ndarray, "torch.Tensor"],
    sequence: Optional[str] = None,
    k: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calcola un dizionario completo di metriche diagnostiche su una matrice FCGR
    (sparsità, occupazione, entropia di Shannon, memoria occupata, distribuzione quadranti).
    Supporta sia matrici 2D `(H, W)` sia tensori multi-canale `(C, H, W)`.
    """
    if torch is not None and isinstance(fcgr_matrix, torch.Tensor):
        arr = fcgr_matrix.detach().cpu().numpy()
    else:
        arr = np.asarray(fcgr_matrix)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr_2d = arr[0]
        n_channels = 1
    elif arr.ndim == 3:
        # Per multi-canale valutiamo la mappa fine (ultimo canale) e la copertura combinata
        arr_2d = arr[-1]
        n_channels = int(arr.shape[0])
    else:
        arr_2d = arr
        n_channels = 1

    side = int(arr_2d.shape[-1])
    inferred_k = int(np.log2(side)) if k is None else int(k)
    total_cells = int(arr.size)
    non_zero_cells = int(np.count_nonzero(arr))
    zero_cells = total_cells - non_zero_cells

    # Sparsità spaziale effettiva (percentuale di pixel (y, x) ciechi su tutti i canali)
    if arr.ndim == 3 and arr.shape[0] > 1:
        active_spatial_pixels = int(np.count_nonzero(np.any(arr > 0, axis=0)))
        spatial_sparsity_pct = (1.0 - active_spatial_pixels / float(side * side)) * 100.0
    else:
        active_spatial_pixels = non_zero_cells
        spatial_sparsity_pct = (1.0 - active_spatial_pixels / float( max(1, total_cells) )) * 100.0

    occupancy_pct = (non_zero_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
    sparsity_pct = 100.0 - occupancy_pct

    total_sum = float(arr_2d.sum())
    if total_sum > 0:
        probs = arr_2d[arr_2d > 0].astype(np.float64) / total_sum
        shannon_entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = float(np.log2(side * side))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        shannon_entropy = 0.0
        normalized_entropy = 0.0

    half = side // 2
    denom = total_sum if total_sum > 0 else 1.0
    q_c = float(arr_2d[:half, :half].sum()) / denom * 100.0
    q_g = float(arr_2d[:half, half:].sum()) / denom * 100.0
    q_a = float(arr_2d[half:, :half].sum()) / denom * 100.0
    q_t = float(arr_2d[half:, half:].sum()) / denom * 100.0

    diagnostics: Dict[str, float] = {
        "k": inferred_k,
        "channels": n_channels,
        "side": side,
        "total_cells": total_cells,
        "non_zero_cells": non_zero_cells,
        "zero_cells": zero_cells,
        "active_spatial_pixels": active_spatial_pixels,
        "occupancy_pct": occupancy_pct,
        "sparsity_pct": sparsity_pct,
        "spatial_sparsity_pct": spatial_sparsity_pct,
        "max_value": float(arr.max()) if total_cells > 0 else 0.0,
        "mean_nonzero_value": float(arr[arr > 0].mean()) if non_zero_cells > 0 else 0.0,
        "shannon_entropy_bits": shannon_entropy,
        "normalized_entropy": normalized_entropy,
        "memory_float32_kb": (total_cells * 4) / 1024.0,
        "memory_float16_kb": (total_cells * 2) / 1024.0,
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
            int(np.count_nonzero(arr_2d)) / valid_kmers if valid_kmers > 0 else 0.0
        )

    if elapsed_ms is not None:
        diagnostics["elapsed_ms"] = float(elapsed_ms)

    return diagnostics


def plot_fcgr_diagnostics(
    sequence_or_fcgr: Union[str, np.ndarray, "torch.Tensor"],
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
        if torch is not None and isinstance(sequence_or_fcgr, torch.Tensor):
            fcgr_norm = sequence_or_fcgr.detach().cpu().numpy()
        else:
            fcgr_norm = np.asarray(sequence_or_fcgr, dtype=np.float32)
        if fcgr_norm.ndim == 3:
            fcgr_norm = fcgr_norm[-1]
        if fcgr_norm.sum() > 1.05:
            fcgr_norm = normalize_fcgr(fcgr_norm)

    diag = compute_fcgr_diagnostics(
        fcgr_norm, sequence=sequence, k=k, elapsed_ms=elapsed_ms
    )

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    im0 = axes[0].imshow(
        fcgr_norm, cmap="viridis", origin="lower", interpolation="nearest"
    )
    axes[0].set_title(f"FCGR Lineare (k={int(diag['k'])}, {int(diag['side'])}x{int(diag['side'])})")
    axes[0].set_xlabel("X (C -> G / A -> T)")
    axes[0].set_ylabel("Y (C -> A / G -> T)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

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
    axes[1].set_title("FCGR Scala Log (esalta k-mer rari)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    occupancy_mask = (fcgr_norm > 0).astype(np.int32)
    im2 = axes[2].imshow(
        occupancy_mask,
        cmap="Greens",
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Mappa Occupazione ({int(diag['non_zero_cells'])}/{int(diag['total_cells'])} celle)\n"
        f"Sparsità: {diag['sparsity_pct']:.2f}%"
    )
    axes[2].set_xlabel("Bianco = 0 (Cella Vuota) | Verde = Attiva")
    axes[2].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])

    axes[3].axis("off")
    seq_info = (
        f"Lunghezza Seq.   : {int(diag['sequence_length']):,} bp\n"
        f"Contenuto GC     : {diag['gc_content_pct']:.2f}%\n"
        f"Unique k-mer Rat.: {diag['unique_ratio']:.3f}\n"
        if "sequence_length" in diag
        else ""
    )
    time_info = (
        f"Tempo Calcolo    : {diag['elapsed_ms']:.3f} ms\n"
        if "elapsed_ms" in diag
        else ""
    )

    summary_text = (
        f"=== DIAGNOSTICA FCGR ===\n"
        f"Sample ID        : {sample_id}\n"
        f"Classe / Label   : {disease}\n"
        f"Ordine k         : {int(diag['k'])} ({int(diag['side'])}x{int(diag['side'])})\n"
        f"{seq_info}"
        f"------------------------\n"
        f"Celle Totali     : {int(diag['total_cells']):,}\n"
        f"Celle Attive (>0): {int(diag['non_zero_cells']):,} ({diag['occupancy_pct']:.2f}%)\n"
        f"Celle Vuote (=0) : {int(diag['zero_cells']):,} ({diag['sparsity_pct']:.2f}%)\n"
        f"Entropia Norm.   : {diag['normalized_entropy']:.3f}\n"
        f"RAM (float32)    : {diag['memory_float32_kb']:.1f} KB\n"
        f"RAM (uint8)      : {diag['memory_uint8_kb']:.1f} KB\n"
        f"{time_info}"
        f"------------------------\n"
        f"Densità Quadranti CGR:\n"
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


def plot_multik_progression(
    sequence: str,
    ks: Sequence[int] = (4, 5, 6, 7, 8),
    sample_id: str = "Sample",
    label_name: str = "Hotspot (1)",
    figsize: Tuple[int, int] = (20, 4.3),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualizza la STESSA sequenza eccDNA al crescere di k (es. k = 4, 5, 6, 7, 8)
    per illustrare nella tesi il trade-off tra precisione spaziale (2^k x 2^k) ed
    esplosione della sparsità (% di zeri) e della memoria.
    """
    fig, axes = plt.subplots(1, len(ks), figsize=figsize)
    if len(ks) == 1:
        axes = [axes]

    for ax, k_val in zip(axes, ks):
        t0 = time.perf_counter()
        counts, _, distinct = compute_fcgr_vectorized(sequence, k_val, return_distinct=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        norm = normalize_fcgr(counts)
        side = 1 << k_val
        total_cells = side * side
        sparsity = (1.0 - distinct / float(total_cells)) * 100.0
        mem_kb = (total_cells * 4) / 1024.0

        im = ax.imshow(
            np.log1p(norm * 1000.0),
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(
            f"k = {k_val} ({side}x{side})\n"
            f"Attive: {distinct}/{total_cells:,}\n"
            f"Sparsità: {sparsity:.1f}% | {mem_kb:.0f} KB",
            fontsize=10,
        )
        ax.set_xlabel(f"Vec: {elapsed_ms:.2f} ms")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Progressione Risoluzione vs Sparsità FCGR sulla Stessa Sequenza — {sample_id} [{label_name}, L={len(sequence):,} bp]",
        fontsize=12.5,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_multiscale_fcgr_diagnostics(
    sequence: str,
    ks: Tuple[int, int, int] = (4, 5, 6),
    target_k: int = 6,
    sample_id: str = "Sample",
    label_name: str = "Hotspot (1)",
    figsize: Tuple[int, int] = (20, 4.5),
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Visualizza il confronto diretto tra la FCGR Baseline (k=6, 1 canale) e la nuova
    FCGR Multi-Scala Ottimizzata (Canale k=4 denso, Canale k=5 medio, Canale k=6 fine +
    fusione RGB multi-canale).
    """
    t0 = time.perf_counter()
    c_base, _, _ = compute_fcgr(sequence, target_k, return_distinct=True)
    f_base = normalize_fcgr(c_base)
    t_base_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    ms_stack, _ = compute_multiscale_fcgr(
        sequence,
        ks=ks,
        target_k=target_k,
        include_reverse_complement=True,
        log_normalize=True,
    )
    ms_u8 = quantize_fcgr_uint8(ms_stack)
    t_opt_ms = (time.perf_counter() - t1) * 1000.0

    d_base = compute_fcgr_diagnostics(f_base, sequence=sequence, k=target_k, elapsed_ms=t_base_ms)
    d_opt = compute_fcgr_diagnostics(ms_stack, sequence=sequence, k=target_k, elapsed_ms=t_opt_ms)

    fig, axes = plt.subplots(1, 5, figsize=figsize)

    # 1. Baseline k=6
    im0 = axes[0].imshow(np.log1p(f_base * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(
        f"Baseline k={target_k} (1ch float32)\nSparsità: {d_base['sparsity_pct']:.1f}% | 16 KB",
        fontsize=9.5,
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2, 3, 4: I 3 canali della Multi-Scala
    cmaps = ["plasma", "viridis", "magma"]
    for idx, k_val in enumerate(ks):
        ch = ms_stack[idx]
        ch_sparsity = (1.0 - np.count_nonzero(ch) / float(ch.size)) * 100.0
        im_c = axes[idx + 1].imshow(ch, cmap=cmaps[idx % len(cmaps)], origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
        axes[idx + 1].set_title(
            f"Canale {idx}: k={k_val} (RC+Log)\n({1<<k_val}x{1<<k_val} -> 64x64) | Spars.: {ch_sparsity:.1f}%",
            fontsize=9.5,
        )
        plt.colorbar(im_c, ax=axes[idx + 1], fraction=0.046, pad=0.04)

    # 5. Composito RGB (R=k4, G=k5, B=k6)
    rgb = np.transpose(ms_stack[:3], (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    axes[4].imshow(rgb, origin="lower", interpolation="nearest")
    axes[4].set_title(
        f"Stack Multi-Scala RGB (3ch uint8)\n"
        f"Sparsità Spaziale: {d_opt['spatial_sparsity_pct']:.1f}% | {ms_u8.nbytes / 1024:.0f} KB",
        fontsize=9.5,
    )
    axes[4].set_xlabel(f"R=k{ks[0]} (Denso), G=k{ks[1]}, B=k{ks[2]} (Fine)")

    plt.suptitle(
        f"FCGR Baseline vs Multi-Scala Frattale Ottimizzata — {sample_id} [{label_name}, L={len(sequence):,} bp] "
        f"| Speedup: {t_base_ms / max(t_opt_ms, 1e-6):.1f}x",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return {
        "baseline_sparsity_pct": d_base["sparsity_pct"],
        "multiscale_spatial_sparsity_pct": d_opt["spatial_sparsity_pct"],
        "baseline_time_ms": t_base_ms,
        "multiscale_time_ms": t_opt_ms,
        "baseline_kb": 16.0,
        "multiscale_uint8_kb": ms_u8.nbytes / 1024.0,
    }


def plot_opt3_enriched_diagnostics(
    sequence: str,
    target_k: int = 6,
    sample_id: str = "Sample",
    label_name: str = "Hotspot (1)",
    figsize: Tuple[int, int] = (20, 4.5),
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Visualizza il pannello a 5 viste per `Opt 3` (Enriched & Balanced Multi-Scale, `sum = 1.0` per canale):
      1. Baseline k=6 (`1ch float32`, `16 KB`)
      2. Canale 0: `k=6 Forward Log-Prob` (`sum = 1.0`, dettaglio fine, senza `/max_val` e senza RC fuso)
      3. Canale 1: `k=6 O/E Motif Enrichment` (`sum = 1.0`, arricchimento sovra-casuale contro il rumore GC/AT)
      4. Canale 2: `Multi-Scala Frattale k=4+5` (`sum = 1.0`, contesto denso bilanciato in norma)
      5. Stack Composito RGB Bilanciato (`3ch uint8`, `12 KB`, -25% RAM, sparsità spaziale <20%)
    """
    t0 = time.perf_counter()
    c_base, _, _ = compute_fcgr(sequence, target_k, return_distinct=True)
    f_base = normalize_fcgr(c_base)
    t_base_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    opt3_stack, _ = compute_opt3_enriched_fcgr(sequence, target_k=target_k)
    opt3_u8 = quantize_fcgr_uint8(opt3_stack)
    opt3_rec = dequantize_fcgr_uint8_prob(opt3_u8)
    t_opt_ms = (time.perf_counter() - t1) * 1000.0

    d_base = compute_fcgr_diagnostics(f_base, sequence=sequence, k=target_k, elapsed_ms=t_base_ms)
    d_opt = compute_fcgr_diagnostics(opt3_rec, sequence=sequence, k=target_k, elapsed_ms=t_opt_ms)

    fig, axes = plt.subplots(1, 5, figsize=figsize)

    im0 = axes[0].imshow(np.log1p(f_base * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(
        f"Baseline k={target_k} (1ch float32)\nSpars.: {d_base['sparsity_pct']:.1f}% | 16 KB",
        fontsize=9.5,
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    ch_titles = [
        f"Ch0: k={target_k} Fwd Log-Prob (sum=1)",
        f"Ch1: k={target_k} O/E Enrichment (sum=1)",
        f"Ch2: Contesto k={max(1,target_k-2)}+{max(1,target_k-1)} Denso (sum=1)",
    ]
    cmaps = ["viridis", "magma", "plasma"]
    for idx in range(3):
        ch = opt3_rec[idx]
        ch_sp = (1.0 - np.count_nonzero(ch) / float(ch.size)) * 100.0
        im_c = axes[idx + 1].imshow(
            np.log1p(ch * 1000.0),
            cmap=cmaps[idx],
            origin="lower",
            interpolation="nearest",
        )
        axes[idx + 1].set_title(
            f"{ch_titles[idx]}\nSpars.: {ch_sp:.1f}% | L2={np.linalg.norm(ch):.4f}",
            fontsize=9.0,
        )
        plt.colorbar(im_c, ax=axes[idx + 1], fraction=0.046, pad=0.04)

    # Per il display RGB scaliamo visivamente ciascun canale in [0, 1]
    rgb = np.transpose(opt3_u8.astype(np.float32) / 255.0, (1, 2, 0))
    axes[4].imshow(np.clip(rgb, 0.0, 1.0), origin="lower", interpolation="nearest")
    axes[4].set_title(
        f"Opt 3 Stack Bilanciato (3ch uint8)\n"
        f"Spars. Spaz.: {d_opt['spatial_sparsity_pct']:.1f}% | {opt3_u8.nbytes / 1024:.0f} KB",
        fontsize=9.5,
    )
    axes[4].set_xlabel(f"R=k{target_k} LogProb, G=O/E Enrich, B=k{max(1,target_k-2)}+{max(1,target_k-1)} Context")

    plt.suptitle(
        f"Opt 3 (Enriched & Balanced Multi-Scale) — {sample_id} [{label_name}, L={len(sequence):,} bp] "
        f"| Speedup: {t_base_ms / max(t_opt_ms, 1e-6):.1f}x",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return {
        "baseline_sparsity_pct": d_base["sparsity_pct"],
        "opt3_spatial_sparsity_pct": d_opt["spatial_sparsity_pct"],
        "baseline_time_ms": t_base_ms,
        "opt3_time_ms": t_opt_ms,
        "baseline_kb": 16.0,
        "opt3_uint8_kb": opt3_u8.nbytes / 1024.0,
    }


def _to_display_2d(arr: np.ndarray) -> np.ndarray:
    """
    Estrae una vista 2D rappresentativa da un array FCGR `(H, W)` o `(C, H, W)`.
    """
    if arr.ndim == 2:
        return arr
    if arr.shape[0] == 1:
        return arr[0]
    return arr.mean(axis=0)


def _scale_for_imshow(vis: np.ndarray) -> np.ndarray:
    """
    Se la matrice è normalizzata a somma unitaria (`max <= 0.2`), applica `log1p(vis * 1000)`
    per renderla ben leggibile a schermo.
    """
    return np.log1p(vis * 1000.0) if float(vis.max()) <= 0.2 else vis


def plot_siamese_pair_feedback(
    x1: Union[np.ndarray, "torch.Tensor"],
    x2: Union[np.ndarray, "torch.Tensor"],
    target: float,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    embedding_distance: Optional[float] = None,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (16, 4.5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualizza una coppia `(x1, x2)` così come entra nei due rami della Rete Siamese,
    mostrando anche la mappa della differenza assoluta `|x1 - x2|` e i dati di sparsità.
    Supporta sia FCGR a 1 canale che FCGR Multi-Scala a 3 canali.
    """
    arr1 = (
        x1.detach().cpu().numpy()
        if (torch is not None and isinstance(x1, torch.Tensor))
        else np.asarray(x1)
    )
    arr2 = (
        x2.detach().cpu().numpy()
        if (torch is not None and isinstance(x2, torch.Tensor))
        else np.asarray(x2)
    )

    d1 = compute_fcgr_diagnostics(arr1)
    d2 = compute_fcgr_diagnostics(arr2)

    vis1 = _to_display_2d(arr1)
    vis2 = _to_display_2d(arr2)
    diff = np.abs(vis1 - vis2)

    l1_dist = float(np.abs(arr1 - arr2).sum())
    l2_dist = float(np.sqrt(np.sum((arr1 - arr2) ** 2)))

    t_int = int(round(target))
    pair_type = "POSITIVA (Stessa Classe)" if t_int == 1 else "NEGATIVA (Classi Diverse)"

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    disp1 = _scale_for_imshow(vis1)
    disp2 = _scale_for_imshow(vis2)

    im0 = axes[0].imshow(disp1, cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(
        f"Ramo 1: {label1}\nSparsità Spaz.: {d1['spatial_sparsity_pct']:.1f}% ({int(d1['active_spatial_pixels'])} px)",
        fontsize=9.5,
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(disp2, cmap="viridis", origin="lower", interpolation="nearest")
    axes[1].set_title(
        f"Ramo 2: {label2}\nSparsità Spaz.: {d2['spatial_sparsity_pct']:.1f}% ({int(d2['active_spatial_pixels'])} px)",
        fontsize=9.5,
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="magma", origin="lower", interpolation="nearest")
    axes[2].set_title(
        f"Differenza |x1 - x2|\nL1={l1_dist:.3f} | L2={l2_dist:.4f}",
        fontsize=9.5,
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].axis("off")
    if embedding_distance is not None:
        emb_line = f"Distanza Embedding : {embedding_distance:.4f}\n"
        if threshold is not None:
            pred_pos = embedding_distance <= threshold
            is_ok = (pred_pos and t_int == 1) or ((not pred_pos) and t_int == 0)
            esito = "CORRETTA" if is_ok else "ERRORE"
            emb_line += f"Soglia Decisionale : {threshold:.4f} -> {esito}\n"
    else:
        emb_line = "Distanza Embedding : (pre-modello)\n"

    info_text = (
        f"=== COPPIA INPUT RETE SIAMESE ===\n"
        f"Tipo Coppia : {pair_type}\n"
        f"Target y    : {t_int} | Canali: {int(d1['channels'])}\n"
        f"---------------------------------\n"
        f"Ramo 1      : {label1}\n"
        f"  Pixel > 0 : {int(d1['active_spatial_pixels'])} / {int(d1['side']*d1['side'])}\n"
        f"  Sparsità  : {d1['spatial_sparsity_pct']:.2f}%\n"
        f"Ramo 2      : {label2}\n"
        f"  Pixel > 0 : {int(d2['active_spatial_pixels'])} / {int(d2['side']*d2['side'])}\n"
        f"  Sparsità  : {d2['spatial_sparsity_pct']:.2f}%\n"
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
        fontsize=9.0,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F6F8", edgecolor="#CBD2D9"),
    )

    plt.suptitle(
        f"Coppia Siamese [{pair_type}] — {label1} vs {label2}",
        fontsize=12.0,
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
    fcgr_baseline: Union[np.ndarray, "torch.Tensor"],
    fcgr_optimized: Union[np.ndarray, "torch.Tensor"],
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
        if (torch is not None and isinstance(fcgr_baseline, torch.Tensor))
        else np.asarray(fcgr_baseline)
    )
    o_arr = (
        fcgr_optimized.detach().cpu().numpy()
        if (torch is not None and isinstance(fcgr_optimized, torch.Tensor))
        else np.asarray(fcgr_optimized)
    )

    b_2d = _to_display_2d(b_arr)
    o_2d = _to_display_2d(o_arr)

    d_base = compute_fcgr_diagnostics(b_arr, elapsed_ms=time_baseline_ms)
    d_opt = compute_fcgr_diagnostics(o_arr, elapsed_ms=time_optimized_ms)

    mem_base_kb = (bytes_baseline / 1024.0) if bytes_baseline else (b_arr.nbytes / 1024.0)
    mem_opt_kb = (bytes_optimized / 1024.0) if bytes_optimized else (o_arr.nbytes / 1024.0)

    speedup = time_baseline_ms / max(time_optimized_ms, 1e-9)
    mem_saving_pct = (1.0 - (mem_opt_kb / max(mem_base_kb, 1e-9))) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(np.log1p(b_2d * 1000.0), cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(
        f"{name_baseline}\nShape: {tuple(b_arr.shape)} | Sparsità Spaz.: {d_base['spatial_sparsity_pct']:.1f}%",
        fontsize=10,
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    disp_opt = o_2d if (o_arr.ndim == 3 and o_arr.shape[0] > 1) else np.log1p(o_2d * 1000.0)
    im1 = axes[1].imshow(disp_opt, cmap="viridis", origin="lower", interpolation="nearest")
    axes[1].set_title(
        f"{name_optimized}\nShape: {tuple(o_arr.shape)} | Sparsità Spaz.: {d_opt['spatial_sparsity_pct']:.1f}%",
        fontsize=10,
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].axis("off")
    same_shape = b_arr.shape == o_arr.shape
    max_abs_err = float(np.max(np.abs(b_arr - o_arr))) if same_shape else float("nan")

    err_line = (
        f"Max Diff Assoluta : {max_abs_err:.2e}\n"
        if same_shape
        else "Rappresentazione Multi-Scala / RC\n"
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
        f"Spars. Spaz. Base : {d_base['spatial_sparsity_pct']:7.2f}%\n"
        f"Spars. Spaz. Opt  : {d_opt['spatial_sparsity_pct']:7.2f}%\n"
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
        "sparsity_baseline_pct": d_base["spatial_sparsity_pct"],
        "sparsity_optimized_pct": d_opt["spatial_sparsity_pct"],
        "max_abs_err": max_abs_err,
    }
