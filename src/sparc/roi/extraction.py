"""ROI extraction with optional parallel processing."""

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

    params = dict(
        min_segment_size        = min_segment_size,
        min_cluster_area        = min_cluster_area,
        min_clean_area          = min_clean_area,
        morph_opening_threshold = morph_opening_threshold,
        max_subclusters         = max_subclusters,
        subcluster_area_divisor = subcluster_area_divisor,
    )

    if use_threading:
        return extract_rois_threaded(
            segmented_img, masked_cube, edge_offset, allowed_variance, n_threads, **params
        )
    return extract_rois_sequential(
        segmented_img, masked_cube, edge_offset, allowed_variance, **params
    )


def extract_rois_sequential(segmented_img, masked_cube, edge_offset, allowed_variance,
                             min_segment_size=50, min_cluster_area=500, min_clean_area=4000,
                             morph_opening_threshold=1000, max_subclusters=10,
                             subcluster_area_divisor=1000):
    from ..utils.geometry import get_roi

    full_mask = masked_cube.mask[0]
    rois      = []

    for seg_id, seg_size in zip(*np.unique(segmented_img, return_counts=True)):
        if seg_size < min_segment_size:
            continue
        result = cluster_segment(
            [segmented_img == seg_id], full_mask, masked_cube,
            edge_offset, allowed_variance,
            min_cluster_area, min_clean_area,
            morph_opening_threshold, max_subclusters, subcluster_area_divisor,
        )
        if result is None:
            continue
        clusters, n_clusters = result
        for cid in range(n_clusters):
            mask = (clusters.data == cid) & ~clusters.mask
            _, roi = get_roi(mask)
            rois.append(roi)

    return np.array(rois) if rois else np.array([])


def extract_rois_threaded(segmented_img, masked_cube, edge_offset, allowed_variance,
                           n_threads, min_segment_size=50, min_cluster_area=500,
                           min_clean_area=4000, morph_opening_threshold=1000,
                           max_subclusters=10, subcluster_area_divisor=1000):
    from ..utils.geometry import get_roi

    full_mask  = masked_cube.mask[0]
    ids, sizes = np.unique(segmented_img, return_counts=True)
    valid      = [(i, s) for i, s in zip(ids, sizes) if s >= min_segment_size]

    def process(seg_info):
        seg_id, _ = seg_info
        result = cluster_segment(
            [segmented_img == seg_id], full_mask, masked_cube,
            edge_offset, allowed_variance,
            min_cluster_area, min_clean_area,
            morph_opening_threshold, max_subclusters, subcluster_area_divisor,
        )
        if result is None:
            return []
        clusters, n_clusters = result
        rois = []
        for cid in range(n_clusters):
            mask = (clusters.data == cid) & ~clusters.mask
            _, roi = get_roi(mask)
            rois.append(roi)
        return rois

    all_rois = [roi for batch in run_parallel(process, valid, n_jobs=n_threads, backend="thread")
                for roi in batch]
    return np.array(all_rois) if all_rois else np.array([])


def cluster_segment(segment_mask, full_mask, spectral_cube, edge_offset, allowed_variance,
                     min_cluster_area, min_clean_area, morph_opening_threshold,
                     max_subclusters, subcluster_area_divisor):
    from ..utils.array_ops import mask_cube

    cluster_mask = prepare_cluster_mask(segment_mask[0], full_mask, edge_offset)
    area         = np.count_nonzero(cluster_mask)

    if area == 0:
        return None
    if area < min_cluster_area:
        return create_single_cluster_result(cluster_mask)

    cluster_mask = clean_mask(cluster_mask, area, morph_opening_threshold)
    area         = np.count_nonzero(cluster_mask)

    if area < min_clean_area:
        return create_single_cluster_result(cluster_mask)

    max_k = max(2, min(max_subclusters, area // subcluster_area_divisor))
    return find_optimal_clusters(mask_cube(spectral_cube, ~cluster_mask), max_k, allowed_variance)


def prepare_cluster_mask(segment_mask: np.ndarray,
                          full_mask: np.ndarray,
                          edge_offset: int) -> np.ndarray:
    from ..utils.geometry import get_edge_mask
    mask = segment_mask.copy()
    mask[full_mask] = 0
    return mask & get_edge_mask(mask.shape, edge_offset)


def clean_mask(mask: np.ndarray, area: int, threshold: int) -> np.ndarray:
    if area > threshold:
        return binary_opening(mask, structure=np.ones((3, 3)))
    return mask


def create_single_cluster_result(mask: np.ndarray) -> Tuple:
    return np.ma.masked_array(np.zeros_like(mask, dtype=np.int32), mask=~mask), 1


def find_optimal_clusters(masked_img: np.ma.MaskedArray,
                           max_clusters: int,
                           allowed_variance: float) -> Tuple:
    """
    Increment k until variance exceeds the threshold or we hit max_clusters.
    Always returns a valid result - k=1 is the floor.
    """
    from ..utils.array_ops import apply_kmeans_to_masked

    best = apply_kmeans_to_masked(masked_img, 1)
    for k in range(2, max_clusters + 1):
        candidate = apply_kmeans_to_masked(masked_img, k)
        if np.var(candidate) >= allowed_variance:
            break
        best = candidate

    return best, k - 1