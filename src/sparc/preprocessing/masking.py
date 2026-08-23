"""Shadow and sky masking for SPARC preprocessing."""

from typing import Any, Dict, TypedDict

import numpy as np
from marslab.imgops.masking import skymask, threshold_mask


class MaskResult(TypedDict):
    masked_cube:      np.ndarray
    shadow_mask:      np.ndarray
    sky_mask:         np.ndarray
    full_mask:        np.ndarray
    left_shadow_mask: np.ndarray
    left_sky_mask:    np.ndarray
    left_full_mask:   np.ndarray
    right_shadow_mask: np.ndarray
    right_sky_mask:   np.ndarray
    right_full_mask:  np.ndarray


def apply_masking(load_result: Dict[str, Any],
                  using_pixmaps: bool,
                  shadow_params: Dict[str, Any],
                  sky_params: Dict[str, Any]) -> MaskResult:
    """Compute shadow and sky masks and apply them to the hyperspectral cube."""
    cube       = load_result['cube']
    left_cube  = load_result['left_cube']
    right_cube = load_result['right_cube']
    instrument = load_result.get('instrument', 'ZCAM')

    if using_pixmaps and instrument == 'ZCAM':
        cube_for_masking  = _right_bands_from_base(load_result['base_bands'])
        left_for_masking  = np.array([a for b, a in load_result['base_bands'].items() if b.startswith('L')])
        right_for_masking = np.array([a for b, a in load_result['base_bands'].items() if b.startswith('R')])
    else:
        cube_for_masking  = right_cube if len(right_cube) >= 3 else cube
        left_for_masking  = left_cube
        right_for_masking = right_cube

    shadow_mask = threshold_mask(cube_for_masking,  **shadow_params)
    # marslab's percentile stretch expects invalid values to be represented
    # by a mask. Plain ndarrays containing even one NaN normalize to all-NaN.
    sky_mask    = _skymask_invalid(cube_for_masking, **sky_params)

    left_shadow_mask  = threshold_mask(left_for_masking,  **shadow_params)
    left_sky_mask     = _skymask_invalid(left_for_masking, **sky_params)
    right_shadow_mask = threshold_mask(right_for_masking, **shadow_params)
    right_sky_mask    = _skymask_invalid(right_for_masking, **sky_params)

    full_mask       = shadow_mask | sky_mask | load_result['homography_mask']
    left_full_mask  = left_shadow_mask  | left_sky_mask
    right_full_mask = right_shadow_mask | right_sky_mask

    masked_cube      = mask_cube(cube, full_mask)
    masked_cube.mask = masked_cube.mask | ~np.isfinite(masked_cube)

    return {
        'masked_cube':      masked_cube,
        'shadow_mask':      shadow_mask,
        'sky_mask':         sky_mask,
        'full_mask':        full_mask,
        'left_shadow_mask': left_shadow_mask,
        'left_sky_mask':    left_sky_mask,
        'left_full_mask':   left_full_mask,
        'right_shadow_mask': right_shadow_mask,
        'right_sky_mask':   right_sky_mask,
        'right_full_mask':  right_full_mask,
    }


def _skymask_invalid(arrays: np.ndarray, **params) -> np.ndarray:
    """Run sky detection while excluding NaN and infinite pixels."""
    return skymask(np.ma.masked_invalid(arrays), **params)


def _right_bands_from_base(base_bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract right-camera bands from base_bands for pixmap-aware masking."""
    return np.array([a for b, a in base_bands.items() if b.startswith('R')])


def mask_cube(cube: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    """Broadcast a 2D mask across all bands of a hyperspectral cube."""
    return np.ma.masked_array(cube, mask=np.repeat(mask[np.newaxis, :], cube.shape[0], axis=0))
