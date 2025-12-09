"""Backend selection and dispatching for SPARC pipeline."""

import numpy as np
import logging
from typing import Callable

from .config import SegmentationBackend, ROIBackend
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def select_segmentation_backend(backend: SegmentationBackend) -> Callable:
    """Select segmentation implementation based on backend."""
    from ..segmentation.sam_segmentation import segment_image
    logger.info(f"Using segmentation backend: {backend.value}")
    return segment_image


def select_roi_backend(backend: ROIBackend) -> Callable:
    """Select ROI extraction implementation based on backend."""
    from ..roi.extraction import extract_rois
    logger.info(f"Using ROI extraction backend: {backend.value}")
    return extract_rois


def dispatch_segmentation(model_path: str,
                         img: np.ndarray,
                         backend: SegmentationBackend,
                         **kwargs) -> np.ndarray:
    """Dispatch segmentation to unified interface."""
    from ..segmentation.sam_segmentation import segment_image
    
    use_gpu = backend != SegmentationBackend.CPU
    
    return segment_image(
        model_path=model_path,
        img=img,
        model_type=kwargs.get('model_type'),
        use_gpu=use_gpu,
        preserve_background=kwargs.get('preserve_background', False),
        points_per_side=kwargs.get('points_per_side', 32),
        pred_iou_thresh=kwargs.get('pred_iou_thresh', 0.88)
    )


def dispatch_roi_extraction(segmented_img: np.ndarray,
                           masked_cube: np.ndarray,
                           edge_offset: int,
                           allowed_variance: float,
                           backend: ROIBackend,
                           **kwargs) -> np.ndarray:
    """
    Dispatch ROI extraction to unified interface.
    
    Args:
        segmented_img: Segmented image
        masked_cube: Masked hyperspectral cube
        edge_offset: Edge offset
        allowed_variance: Allowed variance
        backend: Backend to use
        **kwargs: Additional backend-specific arguments
                 (including min_cluster_area, min_clean_area, etc.)
        
    Returns:
        Array of ROI coordinates
    """
    from ..roi.extraction import extract_rois
    
    use_threading = backend == ROIBackend.THREADED
    
    return extract_rois(
        segmented_img=segmented_img,
        masked_cube=masked_cube,
        edge_offset=edge_offset,
        allowed_variance=allowed_variance,
        use_threading=use_threading,
        n_threads=kwargs.get('n_threads'),
        min_segment_size=kwargs.get('min_segment_size', 50),
        # Pass through advanced clustering params from kwargs
        min_cluster_area=kwargs.get('min_cluster_area', 500),
        min_clean_area=kwargs.get('min_clean_area', 4000),
        morph_opening_threshold=kwargs.get('morph_opening_threshold', 1000),
        max_subclusters=kwargs.get('max_subclusters', 10),
        subcluster_area_divisor=kwargs.get('subcluster_area_divisor', 1000)
    )