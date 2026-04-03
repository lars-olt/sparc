"""Pipeline state dataclass for SPARC."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class SparcState:
    """Complete pipeline state for SPARC processing."""

    # Instrument configuration (set during load_step)
    instrument_config: Optional[Dict[str, Any]] = None

    # Loading stage
    load_result: Optional[Dict[str, Any]] = None
    using_pixmaps: bool = False

    # Preprocessing stage - merged cube
    processed_data: Optional[np.ndarray] = None
    photometrically_calibrated: Optional[np.ndarray] = None
    shadow_mask: Optional[np.ndarray] = None
    sky_mask: Optional[np.ndarray] = None
    full_mask: Optional[np.ndarray] = None

    # Preprocessing stage - left camera
    left_shadow_mask: Optional[np.ndarray] = None
    left_sky_mask: Optional[np.ndarray] = None
    left_full_mask: Optional[np.ndarray] = None

    # Preprocessing stage - right camera
    right_shadow_mask: Optional[np.ndarray] = None
    right_sky_mask: Optional[np.ndarray] = None
    right_full_mask: Optional[np.ndarray] = None

    # Segmentation stage
    segments: Optional[np.ndarray] = None

    # ROI extraction stage
    unfiltered_rois: Optional[np.ndarray] = None
    area_filtered_rois: Optional[np.ndarray] = None
    roi_spectra: Optional[np.ndarray] = None
    roi_stds: Optional[np.ndarray] = None

    # Albedo filtering stage
    albedo_valid_indices: Optional[np.ndarray] = None
    albedo_filtered_spectra: Optional[np.ndarray] = None
    albedo_filtered_stds: Optional[np.ndarray] = None

    # Spectral analysis stage
    outlier_mask: Optional[np.ndarray] = None
    clustering_result: Optional[Dict[str, Any]] = None
    all_clustering_results: Optional[Dict[str, Any]] = None

    # Selection stage
    roi_indices: Optional[List[int]] = None
    final_rois: Optional[np.ndarray] = None       # right-camera space
    final_left_rois: Optional[np.ndarray] = None  # left-camera inscribed rects
    final_spectra: Optional[np.ndarray] = None
    final_stds: Optional[np.ndarray] = None