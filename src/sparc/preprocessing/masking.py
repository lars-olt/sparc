"""Masking functionality for preprocessing."""

import numpy as np
from typing import Dict, Any
from marslab.imgops.masking import skymask, threshold_mask

from ..data.loading import LoadResult
from ..utils.array_ops import mask_cube


# -----------------------------------------------------------------------------
# Huge thanks to Michael St. Clair (@m-stclair)!
# -----------------------------------------------------------------------------


def apply_masking(
    load_result: LoadResult,
    using_pixmaps: bool,
    shadow_kwargs: Dict[str, Any],
    skymask_kwargs: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Apply shadow and sky masking to the data cube.

    Args:
        load_result: Loaded data result
        using_pixmaps: Whether pixmaps were applied during loading
        shadow_kwargs: Parameters for shadow threshold masking
        skymask_kwargs: Parameters for sky masking

    Returns:
        Dictionary containing masked cube and individual masks
    """
    cube = load_result["cube"]

    if using_pixmaps:
        # Use unmasked cube for shadow/sky masking to avoid NaNs
        # NaNs are problematic for masking algorithms
        base_bands = load_result["base_bands"]
        cube_for_masking = np.array(
            [a for b, a in base_bands.items() if b.startswith("R")]
        )
    else:
        # Cube does not contain NaNs when not using pixmaps
        cube_for_masking = cube

    # Generate masks for shadows and sky
    shadow_tmask = threshold_mask(cube_for_masking, **shadow_kwargs)
    sky_tmask = skymask(cube_for_masking, **skymask_kwargs)

    # Combine all masks
    feature_mask = np.logical_or(shadow_tmask, sky_tmask)
    full_tmask = np.logical_or(feature_mask, load_result["homography_tmask"])

    # Apply mask to cube
    cube_preprocessed = mask_cube(cube, full_tmask)

    # Mask any non-finite values
    cube_preprocessed.mask = cube_preprocessed.mask | ~np.isfinite(cube_preprocessed)

    return {
        "masked_cube": cube_preprocessed,
        "shadow_mask": shadow_tmask,
        "sky_mask": sky_tmask,
        "full_mask": full_tmask,
    }
