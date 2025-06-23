"""Array operations and manipulation utilities."""

import numpy as np
from typing import Tuple

from .threading import SafeKMeans


def mask_cube(cube: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    """
    Apply mask to hyperspectral cube.

    Args:
        cube: Hyperspectral data cube
        mask: 2D boolean mask to apply

    Returns:
        Masked array with mask applied to all bands
    """
    stacked_mask = np.repeat(mask[np.newaxis, :], cube.shape[0], axis=0)
    masked_cube = np.ma.masked_array(cube, mask=stacked_mask)
    return masked_cube


def uncompress_cube(
    compressed_data: np.ndarray, pixel_locations: np.ndarray, shape: Tuple[int, ...]
) -> np.ma.MaskedArray:
    """
    Remap compressed data to masked array with original shape.

    Args:
        compressed_data: Compressed data array
        pixel_locations: Valid pixel locations
        shape: Target shape for reconstruction

    Returns:
        Reconstructed masked array
    """
    reconstructed = np.ma.masked_all(shape, dtype=compressed_data.dtype)
    is_cube = len(shape) == 3

    if is_cube:
        bands, _, _ = shape
        pixel_indices = tuple(pixel_locations.T)
        for band in range(bands):
            reconstructed[band][pixel_indices] = compressed_data[band]
    else:
        pixel_indices = tuple(pixel_locations)
        reconstructed[pixel_indices] = compressed_data

    return reconstructed


def apply_kmeans_to_masked(
    masked_array: np.ma.MaskedArray, k: int, seed: int = 42
) -> np.ma.MaskedArray:
    """
    Apply k-means clustering to masked array.

    Args:
        masked_array: Masked hyperspectral data
        k: Number of clusters
        seed: Random seed for reproducibility

    Returns:
        Uncompressed classification results
    """
    # Compress array to contain only unmasked values
    spatial_mask = ~masked_array.mask.any(axis=0)
    valid_pixels = masked_array[:, spatial_mask].data  # Get valid pixels per band
    compressed_cube = valid_pixels.T.astype(np.float32)  # Reshape to (pixels, bands)

    # Apply k-means clustering
    k_means = SafeKMeans(n_clusters=k, random_state=seed)
    classifications = k_means.fit_predict(compressed_cube)

    # Uncompress classifications to original masked shape
    _, h, w = masked_array.shape
    pixel_indices = np.argwhere(spatial_mask).T
    uncompressed_classifications = uncompress_cube(
        classifications, pixel_indices, (h, w)
    )

    return uncompressed_classifications


def normalize_cube(cube: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize hyperspectral cube.

    Args:
        cube: Input hyperspectral cube
        method: Normalization method ('minmax', 'zscore', 'l2')

    Returns:
        Normalized cube
    """
    if method == "minmax":
        cube_min = np.nanmin(cube, axis=(1, 2), keepdims=True)
        cube_max = np.nanmax(cube, axis=(1, 2), keepdims=True)
        normalized = (cube - cube_min) / (cube_max - cube_min)
    elif method == "zscore":
        cube_mean = np.nanmean(cube, axis=(1, 2), keepdims=True)
        cube_std = np.nanstd(cube, axis=(1, 2), keepdims=True)
        normalized = (cube - cube_mean) / cube_std
    elif method == "l2":
        cube_norm = np.linalg.norm(cube, axis=0, keepdims=True)
        normalized = cube / (cube_norm + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def compute_spectral_statistics(cube: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Compute basic spectral statistics for hyperspectral cube.

    Args:
        cube: Hyperspectral data cube
        mask: Optional mask to apply

    Returns:
        Dictionary of statistics
    """
    if mask is not None:
        cube = np.ma.masked_array(cube, mask=np.broadcast_to(mask, cube.shape))

    stats = {
        "mean": np.nanmean(cube, axis=(1, 2)),
        "std": np.nanstd(cube, axis=(1, 2)),
        "min": np.nanmin(cube, axis=(1, 2)),
        "max": np.nanmax(cube, axis=(1, 2)),
        "median": np.nanmedian(cube, axis=(1, 2)),
    }

    return stats
