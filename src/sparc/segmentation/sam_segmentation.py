"""SAM-based image segmentation."""

import numpy as np
from ..utils.device import prepare_accelerators, resolve_device

prepare_accelerators()
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import Optional

from ..utils.memory import release_accelerator_memory


def segment_image(model_path: str,
                 img: np.ndarray,
                 model_type: Optional[str] = None,
                 use_gpu: bool = True,
                 preserve_background: bool = False,
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88,
                 device: Optional[str] = None) -> np.ndarray:
    """Segment an RGB image with SAM and return integer labels per pixel."""
    device = resolve_device(device) if device is not None else select_device(use_gpu)
    model_type = detect_model_type(model_path, model_type)

    sam_model = None
    masks = None
    try:
        sam_model = load_sam_model(model_path, model_type, device)
        masks = generate_masks(sam_model, img, points_per_side, pred_iou_thresh)
        return convert_masks_to_segments(
            masks, img.shape[:2], preserve_background
        )
    finally:
        # SAM is loaded per run. Drop its tensors before returning control to
        # the long-lived desktop process, including when CUDA raises an OOM.
        del masks
        del sam_model
        release_accelerator_memory()


def select_device(use_gpu: bool) -> torch.device:
    """Choose an available accelerator when requested, otherwise CPU."""
    return resolve_device('auto' if use_gpu else 'cpu')


def detect_model_type(model_path: str, model_type: Optional[str]) -> str:
    """Return the explicit SAM model type or infer it from the filename."""
    if model_type is not None:
        return model_type
    
    path_lower = model_path.lower()
    if 'vit_h' in path_lower:
        return 'vit_h'
    elif 'vit_l' in path_lower:
        return 'vit_l'
    elif 'vit_b' in path_lower:
        return 'vit_b'
    
    return 'vit_h'


def load_sam_model(model_path: str,
                  model_type: str,
                  device: torch.device) -> object:
    """Load a SAM checkpoint onto the selected device."""
    sam_model = sam_model_registry[model_type](checkpoint=model_path)
    sam_model.to(device=device)
    return sam_model


def generate_masks(sam_model: object,
                  img: np.ndarray,
                  points_per_side: int,
                  pred_iou_thresh: float) -> list:
    """Generate SAM mask records for an RGB image."""
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    
    return mask_generator.generate(img)


def convert_masks_to_segments(masks: list,
                              shape: tuple,
                              preserve_background: bool) -> np.ndarray:
    """Combine SAM masks into one integer label image."""
    segment_array = np.zeros(shape, dtype=np.int32)
    
    start_id = 1 if preserve_background else 0
    
    for i, mask_data in enumerate(masks, start=start_id):
        segment_array[mask_data["segmentation"]] = i
    
    return segment_array
