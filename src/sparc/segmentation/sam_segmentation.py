"""SAM-based image segmentation."""

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def segment_image(model_path: str, img: np.ndarray) -> np.ndarray:
    """
    Segment image using SAM (Segment Anything Model).

    Args:
        model_path: Path to the SAM model weights
        img: RGB image to segment

    Returns:
        Segmentation mask with integer labels for each segment
    """
    # Initialize SAM model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = "vit_h"

    sam_model = sam_model_registry[model_type](checkpoint=model_path)
    sam_model.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # Generate segmentation masks
    output_masks = mask_generator.generate(img)

    # Convert masks to single labeled array
    h, w, _ = img.shape
    segments = np.zeros((h, w), dtype=np.int32)

    for i, mask_data in enumerate(output_masks):
        mask = mask_data["segmentation"]
        segments[mask] = i

    return segments
