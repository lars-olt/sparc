"""Masking functionality for preprocessing."""

import numpy as np
from typing import Dict, Any, TypedDict
from marslab.imgops.masking import skymask, threshold_mask


class MaskResult(TypedDict):
    """Result from masking operations."""
    masked_cube: np.ndarray
    shadow_mask: np.ndarray
    sky_mask: np.ndarray
    full_mask: np.ndarray
    left_shadow_mask: np.ndarray
    left_sky_mask: np.ndarray
    left_full_mask: np.ndarray
    right_shadow_mask: np.ndarray
    right_sky_mask: np.ndarray
    right_full_mask: np.ndarray


def apply_masking(load_result: Dict[str, Any],
                 using_pixmaps: bool,
                 shadow_params: Dict[str, Any],
                 sky_params: Dict[str, Any]) -> MaskResult:
    """
    Apply shadow and sky masking to hyperspectral cube.
    
    Args:
        load_result: Loaded data result
        using_pixmaps: Whether pixmaps were applied during loading
        shadow_params: Parameters for shadow threshold masking
        sky_params: Parameters for sky masking
        
    Returns:
        MaskResult containing masked cube and individual masks
    """
    cube = load_result['cube']
    left_cube = load_result['left_cube']
    right_cube = load_result['right_cube']
    
    if using_pixmaps:
        cube_for_masking = create_unmasked_cube(load_result['base_bands'])
        left_cube_for_masking = np.array([a for b, a in load_result['base_bands'].items() if b.startswith('L')])
        right_cube_for_masking = np.array([a for b, a in load_result['base_bands'].items() if b.startswith('R')])
    else:
        cube_for_masking = cube
        left_cube_for_masking = left_cube
        right_cube_for_masking = right_cube
    
    # Compute masks for combined (right) cube
    shadow_mask = threshold_mask(cube_for_masking, **shadow_params)
    sky_mask = skymask(cube_for_masking, **sky_params)
    
    # Compute masks for left cube
    left_shadow_mask = threshold_mask(left_cube_for_masking, **shadow_params)
    left_sky_mask = skymask(left_cube_for_masking, **sky_params)
    
    # Compute masks for right cube
    right_shadow_mask = threshold_mask(right_cube_for_masking, **shadow_params)
    right_sky_mask = skymask(right_cube_for_masking, **sky_params)
    
    # Combined masks
    feature_mask = np.logical_or(shadow_mask, sky_mask)
    full_mask = np.logical_or(feature_mask, load_result['homography_mask'])
    
    left_feature_mask = np.logical_or(left_shadow_mask, left_sky_mask)
    left_full_mask = left_feature_mask  # No homography mask for original left
    
    right_feature_mask = np.logical_or(right_shadow_mask, right_sky_mask)
    right_full_mask = right_feature_mask  # No homography mask for right
    
    # Apply mask to merged cube
    masked_cube = mask_cube(cube, full_mask)
    masked_cube.mask = masked_cube.mask | ~np.isfinite(masked_cube)
    
    return {
        'masked_cube': masked_cube,
        'shadow_mask': shadow_mask,
        'sky_mask': sky_mask,
        'full_mask': full_mask,
        'left_shadow_mask': left_shadow_mask,
        'left_sky_mask': left_sky_mask,
        'left_full_mask': left_full_mask,
        'right_shadow_mask': right_shadow_mask,
        'right_sky_mask': right_sky_mask,
        'right_full_mask': right_full_mask
    }


def create_unmasked_cube(base_bands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create cube without NaN values for masking algorithms.
    
    NaNs from pixmaps interfere with masking algorithms,
    so we use the original unmasked data.
    
    Args:
        base_bands: Original band data before pixmap masking
        
    Returns:
        Cube containing only right camera bands
    """
    return np.array([a for b, a in base_bands.items() if b.startswith('R')])


def mask_cube(cube: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    """
    Apply 2D mask to all bands of hyperspectral cube.
    
    Args:
        cube: Hyperspectral data cube
        mask: 2D boolean mask
        
    Returns:
        Masked array with mask applied to all bands
    """
    stacked_mask = np.repeat(mask[np.newaxis, :], cube.shape[0], axis=0)
    return np.ma.masked_array(cube, mask=stacked_mask)