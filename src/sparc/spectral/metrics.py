"""Spectral distance metrics and feature extraction."""

import numpy as np
from typing import Tuple


def spectral_angle_distance(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Calculate spectral angle between two spectra in radians.
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        
    Returns:
        Spectral angle in radians (0 = identical, π/2 = orthogonal)
    """
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    
    if norm1 == 0 or norm2 == 0:
        return np.pi / 2
    
    cos_angle = np.dot(spectrum1, spectrum2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)


def spectral_angle_similarity(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Convert spectral angle to similarity score.
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        
    Returns:
        Similarity score (1 = identical, 0 = orthogonal)
    """
    angle = spectral_angle_distance(spectrum1, spectrum2)
    return 1 - (2 * angle / np.pi)


def euclidean_distance(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two spectra.
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(spectrum1 - spectrum2)


def correlation_distance(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Calculate correlation-based distance between spectra.
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        
    Returns:
        Correlation distance (0 = perfect correlation, 2 = perfect anti-correlation)
    """
    correlation = np.corrcoef(spectrum1, spectrum2)[0, 1]
    
    if np.isnan(correlation):
        return 1.0
    
    return 1 - correlation


def compute_roi_spectra(cube: np.ndarray,
                       roi_coords: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate average spectra for ROI rectangles.
    
    Args:
        cube: Hyperspectral data cube (bands, height, width)
        roi_coords: List of rectangle coordinates (x1, y1, x2, y2)
        
    Returns:
        Tuple of (averaged_spectra, standard_deviations)
    """
    spectra = []
    stds = []
    
    for x1, y1, x2, y2 in roi_coords:
        region = cube[:, y1:y2 + 1, x1:x2 + 1]
        
        avg_spectrum = region.mean(axis=(1, 2))
        std_spectrum = region.std(axis=(1, 2))
        
        spectra.append(avg_spectrum)
        stds.append(std_spectrum)
    
    return np.ma.getdata(spectra), np.ma.getdata(stds)