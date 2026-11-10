"""
Modular package for FCGR representation, dataset pair generation,
visual diagnostics, and Siamese neural network evaluation on eccDNA sequences.
"""

from .fcgr import (
    VALID_BASES,
    CGR_CORNERS,
    load_fasta_sequences,
    compute_fcgr,
    normalize_fcgr,
    sequence_to_fcgr_tensor,
    compute_fcgr_diagnostics,
    plot_fcgr_diagnostics,
    plot_siamese_pair_feedback,
    plot_fcgr_comparison,
)
from .dataset import (
    SiamesePairDataset,
    SiamesePairDatasetMemmap,
    RandomSiamesePairDataset,
    BalancedSiamesePairDataset,
    BalancedSemiHardCandidateDataset,
    SingleFCGRDataset,
)
from .models import (
    FCGRCNNEncoderV3,
    FCGRCNNEncoder,
    SiameseV3,
    SiameseNetwork,
    EuclideanContrastiveLoss,
    ContrastiveLoss,
    select_semihard_negatives,
    evaluate_pairwise,
    extract_embeddings,
    compute_topk_similarity_matrix,
)

__all__ = [
    "VALID_BASES",
    "CGR_CORNERS",
    "load_fasta_sequences",
    "compute_fcgr",
    "normalize_fcgr",
    "sequence_to_fcgr_tensor",
    "compute_fcgr_diagnostics",
    "plot_fcgr_diagnostics",
    "plot_siamese_pair_feedback",
    "plot_fcgr_comparison",
    "SiamesePairDataset",
    "SiamesePairDatasetMemmap",
    "RandomSiamesePairDataset",
    "BalancedSiamesePairDataset",
    "BalancedSemiHardCandidateDataset",
    "SingleFCGRDataset",
    "FCGRCNNEncoderV3",
    "FCGRCNNEncoder",
    "SiameseV3",
    "SiameseNetwork",
    "EuclideanContrastiveLoss",
    "ContrastiveLoss",
    "select_semihard_negatives",
    "evaluate_pairwise",
    "extract_embeddings",
    "compute_topk_similarity_matrix",
]

