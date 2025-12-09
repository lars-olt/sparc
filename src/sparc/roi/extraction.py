"""ROI extraction functionality with optional parallel processing."""

import numpy as np
from scipy.ndimage import binary_opening
from typing import Optional, Tuple

from ..utils.threading import run_parallel


def extract_rois(segmented_img: np.ndarray,
                masked_cube: np.ndarray,
                edge_offset: int,
                allowed_variance: float,
                use_threading: bool = False,
                n_threads: Optional[int] = None,
                min_segment_size: int = 50,
                min_cluster_area: int = 500,
                min_clean_area: int = 4000,
                morph_opening_threshold: int = 1000,
                max_subclusters: int = 10,
                subcluster_area_divisor: int = 1000) -> np.ndarray:
    """
    Extract potential regions of interest from segmented image.
    
    Args:
        segmented_img: SAM segmentation results
        masked_cube: Preprocessed masked hyperspectral cube
        edge_offset: Offset from image edges to avoid
        allowed_variance: Maximum variance allowed in clustering
        use_threading: Use parallel processing
        n_threads: Number of worker threads
        min_segment_size: Skip segments smaller than this
        min_cluster_area: Minimum area to attempt sub-clustering
        min_clean_area: Minimum area after cleaning to sub-cluster
        morph_opening_threshold: Area threshold to apply morph opening
        max_subclusters: Absolute max sub-clusters per segment
        subcluster_area_divisor: Divisor for density-based clustering
        
    Returns:
        Array of ROI coordinates in (x, y, width, height) format
    """
    params = {
        'min_segment_size': min_segment_size,
        'min_cluster_area': min_cluster_area,
        'min_clean_area': min_clean_area,
        'morph_opening_threshold': morph_opening_threshold,
        'max_subclusters': max_subclusters,
        'subcluster_area_divisor': subcluster_area_divisor
    }

    if use_threading:
        return extract_rois_threaded(
            segmented_img, masked_cube, edge_offset, allowed_variance,
            n_threads, **params
        )
    else:
        return extract_rois_sequential(
            segmented_img, masked_cube, edge_offset, allowed_variance,
            **params
        )


def extract_rois_sequential(segmented_img: np.ndarray,
                           masked_cube: np.ndarray,
                           edge_offset: int,
                           allowed_variance: float,
                           min_segment_size: int = 50,
                           min_cluster_area: int = 500,
                           min_clean_area: int = 4000,
                           morph_opening_threshold: int = 1000,
                           max_subclusters: int = 10,
                           subcluster_area_divisor: int = 1000) -> np.ndarray:
    """Extract ROIs sequentially (single-threaded)."""
    from ..utils.geometry import get_roi
    
    full_mask = masked_cube.mask[0]
    rois = []
    
    segment_ids, segment_sizes = np.unique(segmented_img, return_counts=True)
    
    for segment_id, segment_size in zip(segment_ids, segment_sizes):
        if segment_size < min_segment_size:
            continue
            
        segment_mask = [segmented_img == segment_id]
        
        cluster_result = cluster_segment(
            segment_mask, full_mask, masked_cube,
            edge_offset, allowed_variance,
            min_cluster_area, min_clean_area,
            morph_opening_threshold, max_subclusters, subcluster_area_divisor
        )
        
        if cluster_result is None:
            continue
        
        clusters, n_clusters = cluster_result
        
        for cluster_id in range(n_clusters):
            cluster_mask = (clusters.data == cluster_id) & ~clusters.mask
            _, roi = get_roi(cluster_mask)
            rois.append(roi)
    
    return np.array(rois) if rois else np.array([])


def extract_rois_threaded(segmented_img: np.ndarray,
                         masked_cube: np.ndarray,
                         edge_offset: int,
                         allowed_variance: float,
                         n_threads: Optional[int],
                         min_segment_size: int = 50,
                         min_cluster_area: int = 500,
                         min_clean_area: int = 4000,
                         morph_opening_threshold: int = 1000,
                         max_subclusters: int = 10,
                         subcluster_area_divisor: int = 1000) -> np.ndarray:
    """Extract ROIs using thread-based parallelism."""
    from ..utils.geometry import get_roi
    
    full_mask = masked_cube.mask[0]
    
    # 1. Prepare data
    segment_ids, segment_sizes = np.unique(segmented_img, return_counts=True)
    valid_segments = [
        (seg_id, size) for seg_id, size in zip(segment_ids, segment_sizes)
        if size >= min_segment_size
    ]
    
    # 2. Define worker function
    def process_segment(segment_info):
        segment_id, _ = segment_info
        segment_mask = [segmented_img == segment_id]
        
        cluster_result = cluster_segment(
            segment_mask, full_mask, masked_cube,
            edge_offset, allowed_variance,
            min_cluster_area, min_clean_area,
            morph_opening_threshold, max_subclusters, subcluster_area_divisor
        )
        
        if cluster_result is None:
            return []
        
        clusters, n_clusters = cluster_result
        segment_rois = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = (clusters.data == cluster_id) & ~clusters.mask
            _, roi = get_roi(cluster_mask)
            segment_rois.append(roi)
        
        return segment_rois
    
    # 3. Execute in parallel
    results = run_parallel(
        process_segment, 
        valid_segments, 
        n_jobs=n_threads, 
        backend="thread"
    )
    
    # 4. Flatten results
    all_rois = [roi for roi_list in results for roi in roi_list]
    return np.array(all_rois) if all_rois else np.array([])


def cluster_segment(segment_mask: list,
                   full_mask: np.ndarray,
                   spectral_cube: np.ndarray,
                   edge_offset: int,
                   allowed_variance: float,
                   min_cluster_area: int,
                   min_clean_area: int,
                   morph_opening_threshold: int,
                   max_subclusters: int,
                   subcluster_area_divisor: int) -> Optional[Tuple]:
    """Cluster segment spectrally to find homogeneous sub-regions."""
    from ..utils.geometry import get_edge_mask
    from ..utils.array_ops import mask_cube, apply_kmeans_to_masked
    
    cluster_mask = prepare_cluster_mask(segment_mask[0], full_mask, edge_offset)
    initial_area = np.count_nonzero(cluster_mask)
    
    if initial_area == 0:
        return None
    
    if initial_area < min_cluster_area:
        return create_single_cluster_result(cluster_mask)
    
    cleaned_mask = clean_mask(cluster_mask, initial_area, morph_opening_threshold)
    area = np.count_nonzero(cleaned_mask)
    
    if area < min_clean_area:
        return create_single_cluster_result(cleaned_mask)
    
    masked_img = mask_cube(spectral_cube, ~cluster_mask)
    
    # Calculate density-based max clusters
    density_limit = area // subcluster_area_divisor
    max_clusters = min(max_subclusters, density_limit)
    
    # If density limit is too low (e.g., 0), default to at least 2 if we are here
    max_clusters = max(2, max_clusters)
    
    return find_optimal_clusters(masked_img, max_clusters, allowed_variance)


def prepare_cluster_mask(segment_mask: np.ndarray,
                        full_mask: np.ndarray,
                        edge_offset: int) -> np.ndarray:
    """Prepare mask for clustering by removing shadows and edges."""
    from ..utils.geometry import get_edge_mask
    
    cluster_mask = segment_mask.copy()
    cluster_mask[full_mask] = 0
    
    edge_mask = get_edge_mask(cluster_mask.shape, edge_offset)
    return cluster_mask & edge_mask


def clean_mask(mask: np.ndarray, area: int, threshold: int) -> np.ndarray:
    """Apply morphological cleaning to mask."""
    if area > threshold:
        kernel = np.ones((3, 3))
        return binary_opening(mask, structure=kernel)
    return mask


def create_single_cluster_result(mask: np.ndarray) -> Tuple:
    """Create result for single-cluster segment."""
    cluster_array = np.ma.masked_array(
        np.zeros_like(mask).astype(np.int32),
        mask=~mask
    )
    return cluster_array, 1


def find_optimal_clusters(masked_img: np.ma.MaskedArray,
                         max_clusters: int,
                         allowed_variance: float) -> Tuple:
    """Find optimal number of clusters using variance threshold."""
    from ..utils.array_ops import apply_kmeans_to_masked
    
    k = 1
    prev_classification = None
    
    while k <= max_clusters:
        curr_classification = apply_kmeans_to_masked(masked_img, k)
        variance = np.var(curr_classification)
        
        if variance >= allowed_variance:
            return prev_classification, k - 1
        
        prev_classification = curr_classification
        k += 1
    
    return prev_classification, max_clusters