"""ROI filtering and selection functionality."""

import numpy as np
from typing import List, Tuple

from ..core.constants import WAVELENGTHS, BAYER_CUTOFF_INDEX, LEFT_CUTOFF_INDEX


def filter_by_area(rois: np.ndarray, min_area: int) -> np.ndarray:
    """
    Filter ROIs by minimum area threshold.
    
    Args:
        rois: Array of ROI coordinates (x, y, width, height)
        min_area: Minimum area in pixels
        
    Returns:
        Filtered ROI array
    """
    areas = rois[:, 2] * rois[:, 3]
    valid_indices = areas >= min_area
    return rois[valid_indices]


def filter_by_albedo_ratio(spectra: np.ndarray,
                          stds: np.ndarray,
                          threshold: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter spectra by albedo ratio between left and right cameras.
    
    Removes spectra where right camera albedo is less than threshold
    times the left camera albedo. This helps eliminate edge artifacts.
    
    Args:
        spectra: ROI spectra array (full spectra including Bayer bands)
        stds: Standard deviations
        threshold: Minimum albedo ratio (default: 0.80)
        
    Returns:
        Tuple of (filtered_spectra, filtered_stds, valid_indices)
    """
    # Extract non-Bayer bands (indices 3+)
    non_bayer_spectra = spectra[:, 3:]
    n_non_bayer_bands = non_bayer_spectra.shape[1]
    
    # Get wavelengths for the non-Bayer bands that actually exist in the data
    # WAVELENGTHS[3:] has all possible wavelengths, but data may have fewer
    available_wavelengths = WAVELENGTHS[3:3 + n_non_bayer_bands]
    
    # Sort by wavelength
    sorted_indices = np.argsort(available_wavelengths)
    sorted_nb_spectra = non_bayer_spectra[:, sorted_indices]
    
    # Find cutoff between left and right camera bands
    # This is the index where wavelength switches from left to right
    left_right_cutoff = LEFT_CUTOFF_INDEX - BAYER_CUTOFF_INDEX
    
    # Handle case where we don't have enough bands
    if left_right_cutoff >= n_non_bayer_bands or left_right_cutoff <= 0:
        # Can't compute ratio, return all spectra
        return spectra, stds, np.ones(len(spectra), dtype=bool)
    
    left_albedo = sorted_nb_spectra[:, left_right_cutoff - 1]
    right_albedo = sorted_nb_spectra[:, left_right_cutoff]
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        albedo_ratios = right_albedo / left_albedo
        albedo_ratios = np.nan_to_num(albedo_ratios, nan=1.0, posinf=1.0, neginf=1.0)
    
    valid_mask = albedo_ratios >= threshold
    
    return spectra[valid_mask], stds[valid_mask], valid_mask


def select_representative_rois(rois: np.ndarray,
                              stds: np.ndarray,
                              cluster_labels: np.ndarray) -> List[int]:
    """
    Select most representative ROI for each spectral cluster.
    
    Uses heuristic combining area preference and error minimization:
    - Larger ROIs are preferred (more representative)
    - Lower standard deviation is preferred (more homogeneous)
    
    Args:
        rois: ROI coordinates
        stds: Standard deviations for each ROI
        cluster_labels: Cluster assignment for each ROI
        
    Returns:
        List of indices of selected ROIs
    """
    areas = rois[:, 2] * rois[:, 3]
    mean_stds = np.mean(stds, axis=1)
    
    area_range = np.ptp(areas)
    std_range = np.ptp(mean_stds)
    
    selected_indices = []
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        best_index = select_best_roi_in_cluster(
            cluster_indices, areas, mean_stds, area_range, std_range
        )
        selected_indices.append(best_index)
    
    return selected_indices


def select_best_roi_in_cluster(indices: np.ndarray,
                               areas: np.ndarray,
                               mean_stds: np.ndarray,
                               area_range: float,
                               std_range: float) -> int:
    """
    Select best ROI within a cluster using scoring heuristic.
    
    Args:
        indices: Indices of ROIs in cluster
        areas: All ROI areas
        mean_stds: Mean standard deviations
        area_range: Range of areas for normalization
        std_range: Range of stds for normalization
        
    Returns:
        Index of best ROI
    """
    best_score = -np.inf
    best_index = indices[0]
    
    min_area = np.min(areas)
    min_std = np.min(mean_stds)
    
    for idx in indices:
        area_score = normalize_score(areas[idx], min_area, area_range)
        std_score = normalize_score(mean_stds[idx], min_std, std_range)
        
        score = area_score - std_score
        
        if score > best_score:
            best_score = score
            best_index = idx
    
    return best_index


def normalize_score(value: float, minimum: float, value_range: float) -> float:
    """
    Normalize value to 0-1 range.
    
    Args:
        value: Value to normalize
        minimum: Minimum value in range
        value_range: Range of values
        
    Returns:
        Normalized score
    """
    if value_range > 0:
        return (value - minimum) / value_range
    return 0.0