"""SPARC spectral analysis module - Clustering and outlier detection."""

from .analysis import (
    detect_outlier_spectra,
    compute_high_frequency_ratios,
    cluster_with_bayesian_gmm,
    fit_bayesian_gmm,
    create_sam_features,
    find_optimal_k,
    normalize_spectra,
)
from .metrics import (
    spectral_angle_distance,
    spectral_angle_similarity,
    euclidean_distance,
    correlation_distance,
    compute_roi_spectra,
)

__all__ = [
    # Analysis
    'detect_outlier_spectra',
    'compute_high_frequency_ratios',
    'cluster_with_bayesian_gmm',
    'fit_bayesian_gmm',
    'create_sam_features',
    'find_optimal_k',
    'normalize_spectra',
    
    # Metrics
    'spectral_angle_distance',
    'spectral_angle_similarity',
    'euclidean_distance',
    'correlation_distance',
    'compute_roi_spectra',
]