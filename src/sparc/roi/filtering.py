"""ROI filtering and selection functionality."""

import numpy as np
from typing import List

from ..core.constants import WLS, BAYER_CUTOFF, L_CUTOFF


def filter_rois(unfiltered_rois: np.ndarray, area_threshold: int) -> np.ndarray:
    """
    Filter ROIs by minimum area threshold.

    Args:
        unfiltered_rois: Array of ROI coordinates
        area_threshold: Minimum area in pixels

    Returns:
        Filtered array of ROI coordinates
    """
    valid_indices = np.where(
        [roi[2] * roi[3] >= area_threshold for roi in unfiltered_rois]
    )[0]

    return np.array([unfiltered_rois[i] for i in valid_indices])


def apply_selection_heuristics(
    rois: np.ndarray, roi_stds: np.ndarray, classifications: np.ndarray
) -> List[int]:
    """
    Apply heuristics to select most representative ROI for each spectral class.

    The heuristic combines:
    - Area preference (larger is better)
    - Noise minimization (lower standard deviation is better)

    Args:
        rois: Array of ROI coordinates
        roi_stds: Standard deviations for each ROI
        classifications: Cluster labels for each ROI

    Returns:
        List of indices of selected ROIs
    """
    minimized_roi_indices = []

    # Compute normalization factors
    areas = [coords[2] * coords[3] for coords in rois]
    max_area_diff = np.max(areas) - np.min(areas)

    avg_noise = np.mean(roi_stds, axis=1)
    max_err_diff = np.max(avg_noise) - np.min(avg_noise)

    # Find most representative ROI for each spectral category
    for category in np.unique(classifications):

        indices = np.where(classifications == category)[0]

        best_score = -np.inf
        chosen_idx = indices[0]

        for i in indices:
            curr_area = areas[i]
            area_norm = (
                (curr_area - np.min(areas)) / max_area_diff if max_area_diff > 0 else 0
            )

            noise = avg_noise[i]
            error_norm = (
                (noise - np.min(avg_noise)) / max_err_diff if max_err_diff > 0 else 0
            )

            # Score combines area preference and error minimization
            score = area_norm - error_norm

            if score > best_score:
                best_score = score
                chosen_idx = i

        minimized_roi_indices.append(chosen_idx)

    return minimized_roi_indices


def filter_by_albedo_ratio(
    roi_spectra: np.ndarray, roi_stds: np.ndarray, threshold: float = 0.80
) -> tuple:
    """
    Filter spectra by albedo ratio between left and right cameras.

    Removes spectra where right camera's albedo is <80% of left camera's albedo.

    Args:
        roi_spectra: Array of ROI spectra
        roi_stds: Array of ROI standard deviations
        threshold: Minimum ratio threshold

    Returns:
        Tuple of (filtered_spectra, filtered_stds, valid_indices)
    """
    non_bayer_indices = np.argsort(WLS[3:]) + 3
    non_bayer_spectra = roi_spectra[:, non_bayer_indices]

    # Calculate albedo at cutoff between left and right cameras
    l_cnt = L_CUTOFF - BAYER_CUTOFF
    l_albedo = non_bayer_spectra[:, l_cnt - 1]
    r_albedo = non_bayer_spectra[:, l_cnt]

    albedo_ratios = r_albedo / l_albedo
    valid_indices = albedo_ratios >= threshold

    return roi_spectra[valid_indices], roi_stds[valid_indices], valid_indices
