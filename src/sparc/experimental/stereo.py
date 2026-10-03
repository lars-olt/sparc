"""Dense stereo geometry in original image pixel coordinates (no ML imports)."""

from dataclasses import dataclass

import cv2
import numpy as np


def inscribe(mask):
    """Grow an axis-aligned rectangle inside a footprint, including its holes."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    x0, y0 = int(cols.min()), int(rows.min())
    mask = mask[y0:rows.max() + 1, x0:cols.max() + 1]
    # Padding ensures image edges count as boundaries, even for an all-true mask.
    distance = cv2.distanceTransform(
        np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5,
    )[1:-1, 1:-1]
    top, left = np.unravel_index(distance.argmax(), distance.shape)
    bottom, right = top, left
    while True:
        before = (left, top, right, bottom)
        if left > 0 and mask[top:bottom + 1, left - 1].all():
            left -= 1
        if right + 1 < mask.shape[1] and mask[top:bottom + 1, right + 1].all():
            right += 1
        if top > 0 and mask[top - 1, left:right + 1].all():
            top -= 1
        if bottom + 1 < mask.shape[0] and mask[bottom + 1, left:right + 1].all():
            bottom += 1
        if before == (left, top, right, bottom):
            break
    return tuple(int(v) for v in (x0 + left, y0 + top, right - left + 1, bottom - top + 1))


@dataclass
class DenseStereoMapping:
    # Each field lives on the named eye's grid and points into the other eye.
    left_to_right: np.ndarray
    right_to_left: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray

    def map_point(self, point, source):
        grid, valid = self._field(source)
        x, y = (int(round(v)) for v in point)
        if 0 <= y < valid.shape[0] and 0 <= x < valid.shape[1] and valid[y, x]:
            return tuple(float(v) for v in grid[y, x])
        return None

    def _field(self, source):
        if source == 'left':
            return self.left_to_right, self.left_valid
        if source == 'right':
            return self.right_to_left, self.right_valid
        raise ValueError(f'Unknown camera: {source}')

    def map_rect(self, rect, source):
        """Inscribe in the destination pixels whose correspondences lie in rect.

        Use the reverse map to pull the full footprint, rather than projecting
        four corners or filling across occlusions and low-confidence holes.
        """
        if source not in ('left', 'right'):
            raise ValueError(f'Unknown camera: {source}')
        grid, valid = self._field('right' if source == 'left' else 'left')
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return None
        footprint = (valid & (grid[..., 0] >= x) & (grid[..., 0] < x + w)
                     & (grid[..., 1] >= y) & (grid[..., 1] < y + h))
        return inscribe(footprint)

    def warp_left(self, cube):
        grid = self.right_to_left
        warped = np.array([
            cv2.remap(np.asarray(band, dtype=np.float32), grid[..., 0], grid[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=float('nan'))
            for band in cube
        ])
        warped[:, ~self.right_valid] = np.nan
        return warped

    def cropped(self, rect):
        x, y, w, h = rect
        fields, validity = [], []
        for source in ('left', 'right'):
            grid, valid = self._field(source)
            grid = grid[y:y + h, x:x + w].copy() - np.array([x, y], dtype=np.float32)
            valid = valid[y:y + h, x:x + w].copy()
            valid &= ((grid[..., 0] >= 0) & (grid[..., 0] <= w - 1)
                      & (grid[..., 1] >= 0) & (grid[..., 1] <= h - 1))
            fields.append(grid)
            validity.append(valid)
        return DenseStereoMapping(*fields, *validity)
