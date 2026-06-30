"""Photometric calibration for SPARC - converts IOF to R*."""

import math
from typing import Any

import numpy as np

import logging
logger = logging.getLogger(__name__)


def apply_photometric_calibration(masked_cube: np.ndarray,
                                  bandset_metadata: Any,
                                  apply_r_star: bool) -> np.ndarray:
    """
    Convert IOF to R* via R* = IOF / cos(θ).

    θ is derived from INCIDENCE_ANGLE (ZCAM) or SOLAR_ELEVATION (Pancam).
    Falls back to uncalibrated IOF if the angle cannot be determined.
    """
    if not apply_r_star:
        return masked_cube
    try:
        angle = extract_incidence_angle(bandset_metadata)
    except ValueError as e:
        logger.warning(f"R* calibration skipped: {e}")
        return masked_cube
    return masked_cube / np.cos(np.radians(angle))


def extract_incidence_angle(metadata: Any) -> float:
    """
    Derive an effective incidence angle in degrees from instrument metadata.

    ZCAM:  reads INCIDENCE_ANGLE directly from the bandset DataFrame.
    Pancam: converts SOLAR_ELEVATION from the PDS label using
            theta = (solar_elevation + 90) * 2pi / 360.

    Raises ValueError if the angle cannot be determined.
    """
    # ZCAM - bandset metadata is a DataFrame
    if hasattr(metadata, 'columns'):
        if 'INCIDENCE_ANGLE' not in metadata.columns:
            raise ValueError(
                "INCIDENCE_ANGLE not found in bandset metadata."
            )
        return metadata["INCIDENCE_ANGLE"].unique().mean()

    # Pancam - pdr.Metadata exposes metaget()
    if hasattr(metadata, 'metaget'):
        solar_elev = metadata.metaget('SOLAR_ELEVATION')
        if solar_elev is None:
            raise ValueError("SOLAR_ELEVATION not found in PDS label.")
        val = solar_elev['value'] if isinstance(solar_elev, dict) else solar_elev
        if val is None:
            raise ValueError("SOLAR_ELEVATION value is null in PDS label.")
        return math.degrees((val + 90) * 2 * np.pi / 360)

    # Plain dict - normalized label from _normalise_pcam_label
    if isinstance(metadata, dict):
        solar_elev = (
            metadata.get('SITE_DERIVED_GEOMETRY_PARMS', {}).get('SOLAR_ELEVATION')
            or metadata.get('SOLAR_ELEVATION')
        )
        if solar_elev is None:
            raise ValueError("SOLAR_ELEVATION not found in metadata dict.")
        val = solar_elev['value'] if isinstance(solar_elev, dict) else solar_elev
        if val is None:
            raise ValueError("SOLAR_ELEVATION value is null in metadata dict.")
        return math.degrees((val + 90) * 2 * np.pi / 360)

    raise ValueError(f"Unrecognised metadata type: {type(metadata)}.")