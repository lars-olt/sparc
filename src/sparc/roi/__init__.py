"""SPARC ROI module - Region of interest extraction and filtering."""

from .extraction import (
    extract_rois,
    extract_rois_sequential,
    extract_rois_threaded,
    cluster_segment,
    prepare_cluster_mask,
    clean_mask,
    create_single_cluster_result,
    find_optimal_clusters,
)
from .filtering import (
    filter_by_area,
    filter_by_albedo_ratio,
    select_representative_rois,
    select_best_roi_in_cluster,
    normalize_score,
)

__all__ = [
    # Extraction
    'extract_rois',
    'extract_rois_sequential',
    'extract_rois_threaded',
    'cluster_segment',
    'prepare_cluster_mask',
    'clean_mask',
    'create_single_cluster_result',
    'find_optimal_clusters',
    
    # Filtering
    'filter_by_area',
    'filter_by_albedo_ratio',
    'select_representative_rois',
    'select_best_roi_in_cluster',
    'normalize_score',
]