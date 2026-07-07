"""Geometric utility functions for ROI extraction."""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Optional

from ..core.constants import _EDGE_MASK_CACHE, RGB_BANDS, RGB_ENHANCE_KWARGS


def find_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    distance_map = distance_transform_edt(mask)
    normalized_distance = distance_map / distance_map.max()
    density_map = normalized_distance * mask
    max_density_locations = np.where(density_map == 1)
    row, col = max_density_locations
    return int(row[0]), int(col[0])


def find_largest_rectangle(mask: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int, int, int]:
    row, col = int(center[0]), int(center[1])
    height, width = mask.shape
    left = right = col
    top = bottom = row
    can_expand_left = can_expand_right = can_expand_top = can_expand_bottom = True
    while any([can_expand_left, can_expand_right, can_expand_top, can_expand_bottom]):
        can_expand_left  = (left > 0)          and check_rectangle_valid(mask, top, bottom, left - 1, right)
        can_expand_right = (right < width - 1)  and check_rectangle_valid(mask, top, bottom, left, right + 1)
        can_expand_top   = (top > 0)            and check_rectangle_valid(mask, top - 1, bottom, left, right)
        can_expand_bottom= (bottom < height - 1) and check_rectangle_valid(mask, top, bottom + 1, left, right)
        if can_expand_left:   left   -= 1
        if can_expand_right:  right  += 1
        if can_expand_top:    top    -= 1
        if can_expand_bottom: bottom += 1
    return left, top, right, bottom


def check_rectangle_valid(mask: np.ndarray, top: int, bottom: int, left: int, right: int) -> bool:
    return np.all(mask[top:bottom + 1, left:right + 1] == 1)


def extract_roi(mask: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    center = find_center_of_mass(mask)
    left, top, right, bottom = find_largest_rectangle(mask, center)
    width  = right - left + 1
    height = bottom - top + 1
    return width * height, (left, top, width, height)


def inscribed_rect_in_quad(quad: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Largest axis-aligned rectangle that fits inside a convex quadrilateral.

    The parallax-warped parallelogram of a right-camera ROI in left-camera space
    is a non-rectangular quad. Taking its bounding box would cross feature edges;
    this instead inscribes the largest axis-aligned rect that stays fully inside.

    quad: (4, 2) float array of corner points, any winding order.
    Returns (x, y, w, h) or None if the quad is degenerate.
    """
    if quad is None or quad.shape != (4, 2):
        return None

    # Rasterize the quad into a boolean mask to get the exact interior.
    xs, ys = quad[:, 0], quad[:, 1]
    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))

    if x_max <= x_min or y_max <= y_min:
        return None

    # Local coordinates inside the bounding box.
    local_pts = (quad - np.array([x_min, y_min])).astype(np.float32)
    h, w = y_max - y_min, x_max - x_min

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_pts.reshape(-1, 1, 2).astype(np.int32), 1)

    if not mask.any():
        return None

    # Use the distance-transform inscribed-rect approach on the mask.
    center_local = find_center_of_mass(mask.astype(bool))
    l, t, r, b = find_largest_rectangle(mask.astype(bool), center_local)

    # Convert back to image coordinates.
    return (x_min + l, y_min + t, r - l + 1, b - t + 1)


def right_rect_to_left_inscribed(
    right_rect: Tuple,
    homography_matrix: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Transform a right-camera (x, y, w, h) rect into the largest axis-aligned
    rectangle that fits inside its parallax-warped footprint in left-camera space.
    """
    x, y, w, h = right_rect
    corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    inv_H = cv2.invert(homography_matrix)[1]
    left_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), inv_H).reshape(-1, 2)
    return inscribed_rect_in_quad(left_corners)


def create_edge_mask(shape: Tuple[int, int], offset: int) -> np.ndarray:
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
    return [(x, y, x + w, y + h) for x, y, w, h in rois]


def create_rgb_image(cube: np.ndarray) -> np.ndarray:
    from marslab.imgops.imgutils import enhance_color
    rgb_dict  = {band: cube[i] for i, band in enumerate(RGB_BANDS)}
    rgb_stack  = np.stack([rgb_dict['R'], rgb_dict['G'], rgb_dict['B']], axis=-1)
    rgb_masked = np.ma.masked_invalid(rgb_stack)
    return enhance_color(rgb_masked, **RGB_ENHANCE_KWARGS)


# Aliases for backward compatibility
get_edge_mask       = create_edge_mask
get_roi             = extract_roi
rect_to_plot_coords = convert_to_plot_coords
get_rgb_stretch     = create_rgb_image