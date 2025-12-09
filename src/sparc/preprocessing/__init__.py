"""SPARC preprocessing module - Masking and calibration."""

from .masking import (
    apply_masking,
    MaskResult,
    create_unmasked_cube,
    mask_cube,
)
from .calibration import (
    apply_photometric_calibration,
    extract_incidence_angle,
)

__all__ = [
    'apply_masking',
    'MaskResult',
    'create_unmasked_cube',
    'mask_cube',
    'apply_photometric_calibration',
    'extract_incidence_angle',
]