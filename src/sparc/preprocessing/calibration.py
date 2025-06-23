"""Photometric calibration functionality."""

import numpy as np
from ..data.loading import LoadResult


# -----------------------------------------------------------------------------
# Huge thanks to Michael St. Clair (@m-stclair)!
# -----------------------------------------------------------------------------


def apply_photometric_calibration(
    cube_preprocessed: np.ndarray, load_result: LoadResult, apply_r_star: bool
) -> np.ndarray:
    """
    Apply photometric calibration to convert IOF to R*.

    R* = IOF / cos(θ) where θ is the incidence angle.

    Args:
        cube_preprocessed: Preprocessed masked cube
        load_result: Original load result containing metadata
        apply_r_star: Whether to apply R* calibration

    Returns:
        Photometrically calibrated cube
    """
    if apply_r_star:
        # Extract incidence angle from metadata
        # Note: Using mean() here because ZCAM metadata can have multiple
        # solar elevation values in different coordinate systems
        meta = load_result["bandset"].metadata
        incidence = meta["INCIDENCE_ANGLE"].unique().mean()
        photometric_scaling = np.cos(incidence * 2 * np.pi / 360)
    else:
        photometric_scaling = 1

    return cube_preprocessed / photometric_scaling
