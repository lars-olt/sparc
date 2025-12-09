"""Array operations and manipulation utilities."""

import numpy as np
from typing import Tuple


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


def uncompress_cube(compressed_data: np.ndarray,
                   pixel_locations: np.ndarray,
                   shape: Tuple[int, ...]) -> np.ma.MaskedArray:
    """
    Reconstruct full cube from compressed valid pixels.
    
    Args:
        compressed_data: Compressed data array
        pixel_locations: Valid pixel coordinates
        shape: Target shape for reconstruction
        
    Returns:
        Reconstructed masked array
    """
    reconstructed = np.ma.masked_all(shape, dtype=compressed_data.dtype)
    is_cube = len(shape) == 3
    
    if is_cube:
        n_bands = shape[0]
        pixel_indices = tuple(pixel_locations.T)
        
        for band in range(n_bands):
            reconstructed[band][pixel_indices] = compressed_data[band]
    else:
        pixel_indices = tuple(pixel_locations)
        reconstructed[pixel_indices] = compressed_data
    
    return reconstructed


def apply_kmeans_to_masked(masked_array: np.ma.MaskedArray,
                          n_clusters: int,
                          random_seed: int = 42) -> np.ma.MaskedArray:
    """
    Apply k-means clustering to masked array, handling NaN values.
    
    Args:
        masked_array: Masked hyperspectral data
        n_clusters: Number of clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        Classification results in original shape
    """
    from .threading import SafeKMeans
    
    spatial_mask = ~masked_array.mask.any(axis=0)
    valid_pixels = masked_array[:, spatial_mask].data
    compressed = valid_pixels.T.astype(np.float32)
    
    kmeans = SafeKMeans(n_clusters=n_clusters, random_state=random_seed)
    labels = kmeans.fit_predict(compressed)
    
    _, height, width = masked_array.shape
    pixel_indices = np.argwhere(spatial_mask).T
    
    return uncompress_cube(labels, pixel_indices, (height, width))


def normalize_cube(cube: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize hyperspectral cube.
    
    Args:
        cube: Input cube
        method: Normalization method ('minmax', 'zscore', 'l2')
        
    Returns:
        Normalized cube
    """
    if method == 'minmax':
        return normalize_minmax(cube)
    elif method == 'zscore':
        return normalize_zscore(cube)
    elif method == 'l2':
        return normalize_l2(cube)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_minmax(cube: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range per band."""
    cube_min = np.nanmin(cube, axis=(1, 2), keepdims=True)
    cube_max = np.nanmax(cube, axis=(1, 2), keepdims=True)
    return (cube - cube_min) / (cube_max - cube_min)


def normalize_zscore(cube: np.ndarray) -> np.ndarray:
    """Normalize to zero mean and unit variance per band."""
    cube_mean = np.nanmean(cube, axis=(1, 2), keepdims=True)
    cube_std = np.nanstd(cube, axis=(1, 2), keepdims=True)
    return (cube - cube_mean) / cube_std


def normalize_l2(cube: np.ndarray) -> np.ndarray:
    """Normalize using L2 norm per pixel."""
    cube_norm = np.linalg.norm(cube, axis=0, keepdims=True)
    return cube / (cube_norm + 1e-10)


def compute_spectral_statistics(cube: np.ndarray,
                               mask: np.ndarray = None) -> dict:
    """
    Compute basic spectral statistics for cube.
    
    Args:
        cube: Hyperspectral data cube
        mask: Optional mask to apply
        
    Returns:
        Dictionary of statistics per band
    """
    if mask is not None:
        cube = np.ma.masked_array(cube, mask=np.broadcast_to(mask, cube.shape))
    
    return {
        'mean': np.nanmean(cube, axis=(1, 2)),
        'std': np.nanstd(cube, axis=(1, 2)),
        'min': np.nanmin(cube, axis=(1, 2)),
        'max': np.nanmax(cube, axis=(1, 2)),
        'median': np.nanmedian(cube, axis=(1, 2))
    }