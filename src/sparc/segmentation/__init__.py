"""SPARC segmentation module - SAM-based image segmentation."""

from .sam_segmentation import (
    segment_image,
    select_device,
    detect_model_type,
    load_sam_model,
    generate_masks,
    convert_masks_to_segments,
)

__all__ = [
    'segment_image',
    'select_device',
    'detect_model_type',
    'load_sam_model',
    'generate_masks',
    'convert_masks_to_segments',
]