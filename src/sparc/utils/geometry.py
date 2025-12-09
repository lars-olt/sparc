"""Geometric utility functions for ROI extraction."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple

from ..core.constants import _EDGE_MASK_CACHE, RGB_BANDS


def find_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    """
    Find center of mass using distance transform for density weighting.
    
    Args:
        mask: Boolean mask array
        
    Returns:
        Tuple of (row, col) coordinates
    """
    distance_map = distance_transform_edt(mask)
    normalized_distance = distance_map / distance_map.max()
    density_map = normalized_distance * mask
    
    max_density_locations = np.where(density_map == 1)
    row, col = max_density_locations
    
    return int(row[0]), int(col[0])


def find_largest_rectangle(mask: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Find largest rectangle centered at point that fits within mask.
    
    Args:
        mask: Boolean mask array
        center: Center point (row, col)
        
    Returns:
        Tuple of (left, top, right, bottom) coordinates
    """
    row, col = int(center[0]), int(center[1])
    height, width = mask.shape
    
    left = right = col
    top = bottom = row
    
    can_expand_left = True
    can_expand_right = True
    can_expand_top = True
    can_expand_bottom = True
    
    while any([can_expand_left, can_expand_right, can_expand_top, can_expand_bottom]):
        can_expand_left = (left > 0) and check_rectangle_valid(mask, top, bottom, left - 1, right)
        can_expand_right = (right < width - 1) and check_rectangle_valid(mask, top, bottom, left, right + 1)
        can_expand_top = (top > 0) and check_rectangle_valid(mask, top - 1, bottom, left, right)
        can_expand_bottom = (bottom < height - 1) and check_rectangle_valid(mask, top, bottom + 1, left, right)
        
        if can_expand_left:
            left -= 1
        if can_expand_right:
            right += 1
        if can_expand_top:
            top -= 1
        if can_expand_bottom:
            bottom += 1
    
    return left, top, right, bottom


def check_rectangle_valid(mask: np.ndarray,
                         top: int,
                         bottom: int,
                         left: int,
                         right: int) -> bool:
    """
    Check if rectangle region is fully within mask.
    
    Args:
        mask: Boolean mask
        top: Top coordinate
        bottom: Bottom coordinate
        left: Left coordinate
        right: Right coordinate
        
    Returns:
        True if all pixels in rectangle are masked
    """
    return np.all(mask[top:bottom + 1, left:right + 1] == 1)


def extract_roi(mask: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """
    Extract ROI rectangle for masked region.
    
    Args:
        mask: Boolean mask array
        
    Returns:
        Tuple of (area, rectangle) where rectangle is (left, top, width, height)
    """
    center = find_center_of_mass(mask)
    left, top, right, bottom = find_largest_rectangle(mask, center)
    
    width = right - left + 1
    height = bottom - top + 1
    area = width * height
    
    return area, (left, top, width, height)


def create_edge_mask(shape: Tuple[int, int], offset: int) -> np.ndarray:
    """
    Create edge mask with caching for performance.
    
    Args:
        shape: Image shape (height, width)
        offset: Offset from edges in pixels
        
    Returns:
        Boolean mask with edge pixels set to False
    """
    cache_key = (shape, offset)
    
    if cache_key in _EDGE_MASK_CACHE:
        return _EDGE_MASK_CACHE[cache_key]
    
    height, width = shape
    edge_mask = np.ones(shape, dtype=bool)
    
    edge_mask[:, :offset] = 0
    edge_mask[:, width - offset:] = 0
    edge_mask[:offset, :] = 0
    edge_mask[height - offset:, :] = 0
    
    _EDGE_MASK_CACHE[cache_key] = edge_mask
    return edge_mask


def convert_to_plot_coords(rois: np.ndarray) -> list:
    """
    Convert ROI format from (x, y, w, h) to (x1, y1, x2, y2).
    
    Args:
        rois: Array of rectangles in (x, y, width, height) format
        
    Returns:
        List of rectangles in (x1, y1, x2, y2) format
    """
    plot_coords = []
    
    for x, y, width, height in rois:
        plot_coords.append((x, y, x + width, y + height))
    
    return plot_coords


def create_rgb_image(cube: np.ndarray) -> np.ndarray:
    """
    Create RGB image from hyperspectral cube.
    
    Args:
        cube: Hyperspectral data cube
        
    Returns:
        RGB image array
    """
    from marslab.imgops.imgutils import enhance_color
    
    rgb_dict = {band: cube[i] for i, band in enumerate(RGB_BANDS)}
    rgb_stack = np.stack([rgb_dict['R'], rgb_dict['G'], rgb_dict['B']], axis=-1)
    rgb_masked = np.ma.masked_invalid(rgb_stack)
    
    return enhance_color(rgb_masked, bounds=(0, 1), stretch=0.1)


# Alias for backward compatibility
get_edge_mask = create_edge_mask
get_roi = extract_roi
rect_to_plot_coords = convert_to_plot_coords
get_rgb_stretch = create_rgb_image