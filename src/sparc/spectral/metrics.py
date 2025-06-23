"""Spectral distance metrics and feature extraction."""

import numpy as np
from typing import Optional

from ..utils.threading import SafeKMeans


def spectral_angle_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """
    Calculate spectral angle between two spectra in radians.

    Args:
        spec1: First spectrum
        spec2: Second spectrum

    Returns:
        Spectral angle in radians (0 = identical, π/2 = orthogonal)
    """
    # Ensure no zero vectors
    if np.linalg.norm(spec1) == 0 or np.linalg.norm(spec2) == 0:
        return np.pi / 2

    # Calculate cosine of angle
    cos_angle = np.dot(spec1, spec2) / (np.linalg.norm(spec1) * np.linalg.norm(spec2))

    # Clip to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Return angle in radians
    return np.arccos(cos_angle)


def spectral_angle_similarity(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """
    Convert spectral angle to similarity score.

    Args:
        spec1: First spectrum
        spec2: Second spectrum

    Returns:
        Similarity score (1 = identical, 0 = orthogonal)
    """
    angle = spectral_angle_distance(spec1, spec2)
    return 1 - (2 * angle / np.pi)  # Normalize to 0-1 range


def sam_based_features(
    spectra: np.ndarray,
    num_components: int,
    reference_spectra: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create features based on spectral angles to reference spectra.

    This allows using GMM with SAM-derived features for better clustering
    when spectral angle is the preferred distance metric.

    Args:
        spectra: Input spectra array
        num_components: Number of components for reference generation
        reference_spectra: Optional pre-defined reference spectra

    Returns:
        SAM-based feature array
    """
    if reference_spectra is None:
        # Use k-means centroids as references
        kmeans = SafeKMeans(n_clusters=min(10, num_components), random_state=42)
        kmeans.fit(spectra)
        reference_spectra = kmeans.cluster_centers_

    # Calculate spectral angles to each reference
    sam_features = np.zeros((len(spectra), len(reference_spectra)))

    for i, spectrum in enumerate(spectra):
        for j, ref_spectrum in enumerate(reference_spectra):
            # Use similarity instead of distance for better GMM performance
            sam_features[i, j] = spectral_angle_similarity(spectrum, ref_spectrum)

    return sam_features


def average_spectra(data: np.ndarray, rectangles: list) -> tuple:
    """
    Calculate average spectra for each rectangle region in the hyperspectral cube.

    Args:
        data: Hyperspectral data cube (bands, height, width)
        rectangles: List of rectangle coordinates (x1, y1, x2, y2)

    Returns:
        Tuple of (averaged_spectra, standard_deviations)
    """
    averaged_spectra = []
    std_spectra = []

    for x1, y1, x2, y2 in rectangles:
        # Extract the region within the rectangle
        region = data[:, y1 : y2 + 1, x1 : x2 + 1]

        # Average over the spatial dimensions (height, width)
        avg_spectrum = region.mean(axis=(1, 2))
        averaged_spectra.append(avg_spectrum)

        std_spectrum = region.std(axis=(1, 2))
        std_spectra.append(std_spectrum)

    return np.ma.getdata(averaged_spectra), np.ma.getdata(std_spectra)


def euclidean_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two spectra.

    Args:
        spec1: First spectrum
        spec2: Second spectrum

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(spec1 - spec2)


def correlation_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """
    Calculate correlation-based distance between two spectra.

    Args:
        spec1: First spectrum
        spec2: Second spectrum

    Returns:
        Correlation distance (0 = perfect correlation, 2 = perfect anti-correlation)
    """
    correlation = np.corrcoef(spec1, spec2)[0, 1]
    # Handle NaN correlation (constant spectra)
    if np.isnan(correlation):
        return 1.0
    return 1 - correlation
