"""SAM-based image segmentation with flexible backend options."""

import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import Optional


def segment_image(model_path: str,
                 img: np.ndarray,
                 model_type: Optional[str] = None,
                 use_gpu: bool = True,
                 preserve_background: bool = False,
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88) -> np.ndarray:
    """
    Segment image using SAM with automatic configuration.
    
    This unified function handles all segmentation modes by automatically
    detecting model type and using appropriate settings based on parameters.
    
    Args:
        model_path: Path to SAM model checkpoint
        img: RGB image array (H, W, 3)
        model_type: Model type ('vit_h', 'vit_l', 'vit_b') or None for auto-detect
        use_gpu: Use GPU if available
        preserve_background: If True, unclassified pixels become segment 0
        points_per_side: Number of points per side for mask generation
        pred_iou_thresh: IoU threshold for mask prediction
        
    Returns:
        Segmentation array where each pixel has a segment ID
    """
    device = select_device(use_gpu)
    model_type = detect_model_type(model_path, model_type)
    
    sam_model = load_sam_model(model_path, model_type, device)
    masks = generate_masks(sam_model, img, points_per_side, pred_iou_thresh)
    
    return convert_masks_to_segments(masks, img.shape[:2], preserve_background)


def select_device(use_gpu: bool) -> torch.device:
    """
    Select compute device based on availability and preference.
    
    Args:
        use_gpu: Whether to prefer GPU
        
    Returns:
        Torch device (cuda or cpu)
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def detect_model_type(model_path: str, model_type: Optional[str]) -> str:
    """
    Auto-detect SAM model type from filename if not specified.
    
    Args:
        model_path: Path to model file
        model_type: Explicit model type (overrides detection)
        
    Returns:
        Model type string
    """
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
    """
    Load and prepare SAM model.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model architecture type
        device: Device to load model on
        
    Returns:
        Loaded SAM model
    """
    sam_model = sam_model_registry[model_type](checkpoint=model_path)
    sam_model.to(device=device)
    return sam_model


def generate_masks(sam_model: object,
                  img: np.ndarray,
                  points_per_side: int,
                  pred_iou_thresh: float) -> list:
    """
    Generate segmentation masks using SAM.
    
    Args:
        sam_model: Loaded SAM model
        img: RGB image
        points_per_side: Sampling density
        pred_iou_thresh: Quality threshold
        
    Returns:
        List of mask dictionaries
    """
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
    """
    Convert SAM mask list to single segmentation array.
    
    Args:
        masks: List of mask dictionaries from SAM
        shape: Image shape (height, width)
        preserve_background: Start numbering from 1 (background = 0)
        
    Returns:
        Segmentation array with unique ID per segment
    """
    segment_array = np.zeros(shape, dtype=np.int32)
    
    start_id = 1 if preserve_background else 0
    
    for i, mask_data in enumerate(masks, start=start_id):
        segment_array[mask_data["segmentation"]] = i
    
    return segment_array