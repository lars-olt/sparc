"""ROI extraction functionality."""

import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_opening

from ..utils.array_ops import mask_cube, apply_kmeans_to_masked
from ..utils.geometry import get_edge_mask, get_roi


def get_potential_rois(
    segmented_img: np.ndarray,
    masked_cube: np.ndarray,
    edge_offset: int,
    allowed_variance: float,
) -> np.ndarray:
    """
    Extract potential regions of interest from segmented image.

    Args:
        segmented_img: SAM segmentation results
        masked_cube: Preprocessed masked hyperspectral cube
        edge_offset: Offset from image edges to avoid
        allowed_variance: Maximum variance allowed in clustering

    Returns:
        Array of ROI coordinates in (x, y, width, height) format
    """
    full_tmask = masked_cube.mask[0]  # Same mask copied over all bands

    rois = []

    # Process each SAM segment
    for region_i in range(len(np.unique(segmented_img))):

        # Create region mask for current segment
        region = [segmented_img == region_i]
        cluster_result = cluster_region(
            region, full_tmask, masked_cube, edge_offset, allowed_variance
        )

        if cluster_result is None:
            continue

        clusters, k = cluster_result

        # Extract ROI for each cluster within the segment
        for cluster in range(k):
            cluster_slice = (clusters.data == cluster) & ~clusters.mask
            _, roi = get_roi(cluster_slice)
            rois.append(roi)

    return np.array(rois)


def cluster_region(
    region_mask: list,
    full_mask: np.ndarray,
    spectral_cube: np.ndarray,
    edge_offset: int,
    allowed_variance: float,
):
    """
    Cluster a single region spectrally to find homogeneous sub-regions.

    Args:
        region_mask: List containing boolean mask for the region
        full_mask: Full image mask
        spectral_cube: Hyperspectral data cube
        edge_offset: Edge offset to apply
        allowed_variance: Maximum allowed variance

    Returns:
        Tuple of (classification_result, number_of_clusters) or None
    """
    k = 1
    variance = 0
    prev_classification = []
    k_found = False

    # Remove shadow regions and apply edge offset
    cluster_mask = region_mask[0].copy()
    cluster_mask[full_mask] = 0

    # Apply edge mask
    edge_mask = get_edge_mask(cluster_mask.shape, edge_offset)
    cluster_mask = cluster_mask & edge_mask

    initial_area = np.count_nonzero(cluster_mask)
    if initial_area == 0:
        return None

    # For small regions, treat as single pebble
    if initial_area < 500:
        pebble_mask = np.ma.masked_array(
            np.zeros_like(cluster_mask).astype(np.int32), mask=~cluster_mask
        )
        return pebble_mask, 1

    # Clean mask with morphological operations for larger regions
    area_before_cleaning = np.count_nonzero(cluster_mask)
    if area_before_cleaning > 1000:
        erosion_kernel = (3, 3)
        cleaned_mask = binary_opening(cluster_mask, structure=np.ones(erosion_kernel))
    else:
        cleaned_mask = cluster_mask

    # For moderate size regions, treat as single pebble
    area = np.count_nonzero(cleaned_mask)
    if area < 4000:
        pebble_mask = np.ma.masked_array(
            np.zeros_like(cleaned_mask).astype(np.int32), mask=~cleaned_mask
        )
        return pebble_mask, k

    # Apply mask to spectral data
    masked_img = mask_cube(spectral_cube, ~cluster_mask)

    max_k = min(10, area // 1000)

    # Iteratively increase k until variance exceeds threshold
    while not k_found and k <= max_k:

        curr_classification = apply_kmeans_to_masked(masked_img, k)
        variance = np.var(curr_classification)

        k_found = variance >= allowed_variance
        if not k_found:
            prev_classification = curr_classification
            k += 1
        else:
            k -= 1

    if k > max_k:
        k = max_k
        prev_classification = curr_classification

    return prev_classification, k
