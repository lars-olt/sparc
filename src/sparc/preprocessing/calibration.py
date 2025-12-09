"""Photometric calibration functionality."""

import numpy as np
from typing import Dict, Any


def apply_photometric_calibration(masked_cube: np.ndarray,
                                 bandset_metadata: Dict[str, Any],
                                 apply_r_star: bool) -> np.ndarray:
    """
    Apply photometric calibration to convert IOF to R*.
    
    R* = IOF / cos(θ) where θ is the incidence angle.
    
    Args:
        masked_cube: Preprocessed masked cube
        bandset_metadata: Metadata containing incidence angle
        apply_r_star: Whether to apply R* calibration
        
    Returns:
        Photometrically calibrated cube
    """
    if apply_r_star:
        incidence_angle = extract_incidence_angle(bandset_metadata)
        scaling_factor = np.cos(np.radians(incidence_angle))
    else:
        scaling_factor = 1.0
    
    return masked_cube / scaling_factor


def extract_incidence_angle(metadata: Dict[str, Any]) -> float:
    """
    Extract incidence angle from metadata.
    
    Uses mean because ZCAM metadata can have multiple values
    in different coordinate systems.
    
    Args:
        metadata: Bandset metadata dictionary
        
    Returns:
        Mean incidence angle in degrees
    """
    return metadata["INCIDENCE_ANGLE"].unique().mean()