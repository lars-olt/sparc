"""Geometric utility functions."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple

from ..core.constants import _EDGE_MASK_CACHE, RGB_MAPPING
from marslab.imgops.imgutils import enhance_color


def get_center_of_mass(masked_arr: np.ndarray) -> Tuple[int, int]:
    """
    Find center of mass using distance transform for density weighting.

    Args:
        masked_arr: Boolean mask array

    Returns:
        Tuple of (row, col) coordinates of center of mass
    """
    # Compute density map based on distance from edges
    distance_transform = distance_transform_edt(masked_arr)

    # Normalize distances
    normalized_distance = distance_transform / distance_transform.max()

    # Apply original array as mask (only interested in density within target region)
    density_within_mask = normalized_distance * masked_arr

    # Find highest density location
    highest_density_loc = np.where(density_within_mask == 1)

    # Return first center of mass found
    cxs, cys = highest_density_loc
    center_of_mass = (int(cxs[0]), int(cys[0]))

    return center_of_mass


def largest_rect_around_center(
    mask: np.ndarray, center: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Find the largest rectangle centered at given point that fits within the mask.

    Args:
        mask: Boolean mask array
        center: Center point (row, col)

    Returns:
        Tuple of (left, top, right, bottom) coordinates
    """
    row, col = int(center[0]), int(center[1])
    total_rows, total_cols = mask.shape

    # Initialize boundaries to center point
    left = right = col
    top = bottom = row

    # Track expansion possibilities
    left_inbounds = True
    right_inbounds = True
    top_inbounds = True
    bottom_inbounds = True

    # Expand in each direction until image edge or mask edge is reached
    while left_inbounds or right_inbounds or top_inbounds or bottom_inbounds:
        left_inbounds = (left > 0) and np.all(
            mask[top : bottom + 1, left - 1 : right + 1] == 1
        )
        right_inbounds = (right < total_cols - 1) and np.all(
            mask[top : bottom + 1, left : right + 2] == 1
        )
        top_inbounds = (top > 0) and np.all(
            mask[top - 1 : bottom + 1, left : right + 1] == 1
        )
        bottom_inbounds = (bottom < total_rows - 1) and np.all(
            mask[top : bottom + 2, left : right + 1] == 1
        )

        if left_inbounds:
            left -= 1
        if right_inbounds:
            right += 1
        if top_inbounds:
            top -= 1
        if bottom_inbounds:
            bottom += 1

    return (left, top, right, bottom)


def get_roi(masked_arr: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """
    Extract ROI rectangle for a masked region.

    Args:
        masked_arr: Boolean mask array

    Returns:
        Tuple of (area, rectangle_coords) where rectangle is (left, top, width, height)
    """
    center_of_mass = get_center_of_mass(masked_arr)

    # Find the largest rectangle centered at this point
    left, top, right, bottom = largest_rect_around_center(masked_arr, center_of_mass)

    width = right - left + 1
    height = bottom - top + 1
    area = width * height

    rect = (left, top, width, height)

    return area, rect


def get_edge_mask(shape: Tuple[int, int], edge_offset: int) -> np.ndarray:
    """
    Create edge mask with caching for performance.

    Args:
        shape: Image shape (height, width)
        edge_offset: Offset from edges in pixels

    Returns:
        Boolean mask with edge pixels set to False
    """
    if (shape, edge_offset) in _EDGE_MASK_CACHE:
        return _EDGE_MASK_CACHE[(shape, edge_offset)]

    max_y, max_x = shape
    edge_mask = np.ones(shape, dtype=bool)
    edge_mask[:, :edge_offset] = 0
    edge_mask[:, (max_x - edge_offset) :] = 0
    edge_mask[:edge_offset, :] = 0
    edge_mask[(max_y - edge_offset) :, :] = 0

    _EDGE_MASK_CACHE[(shape, edge_offset)] = edge_mask
    return edge_mask


def rect_to_plot_coords(rect_coords: np.ndarray) -> list:
    """
    Convert rectangle coordinates from (x, y, w, h) to (x1, y1, x2, y2) format.

    Args:
        rect_coords: Array of rectangles in (x, y, width, height) format

    Returns:
        List of rectangles in (x1, y1, x2, y2) format
    """
    plt_coords = []
    for i in range(len(rect_coords)):
        x1, y1, w, h = rect_coords[i]
        x2 = x1 + w
        y2 = y1 + h
        plt_coords.append((x1, y1, x2, y2))
    return plt_coords


def get_rgb_stretch(cube: np.ndarray) -> np.ndarray:
    """
    Create RGB stretched image from hyperspectral cube.

    Args:
        cube: Hyperspectral data cube

    Returns:
        RGB stretched image
    """
    img = {}
    for i, color in enumerate(RGB_MAPPING):
        img[color] = cube[i]

    mapped_img = [img["R"], img["G"], img["B"]]
    rgb = np.ma.masked_invalid(np.stack(mapped_img, axis=-1))
    rgb_stretch = enhance_color(rgb, bounds=(0, 1), stretch=0.1)

    return rgb_stretch
