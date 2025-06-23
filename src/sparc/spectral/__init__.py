"""Spectral analysis functionality."""

from .analysis import detect_unique_spectra, auto_bayesian_gmm, cluster_spectra
from .metrics import spectral_angle_distance, spectral_angle_similarity, average_spectra

__all__ = [
    'detect_unique_spectra', 
    'auto_bayesian_gmm', 
    'cluster_spectra',
    'spectral_angle_distance', 
    'spectral_angle_similarity', 
    'average_spectra'
]