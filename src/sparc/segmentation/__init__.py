"""SPARC SAM segmentation, exposed lazily as an optional feature."""

from .._lazy import public_dir, resolve_attribute


_LAZY_IMPORTS = {
    "segment_image": (".sam_segmentation", "segment_image"),
    "select_device": (".sam_segmentation", "select_device"),
    "detect_model_type": (".sam_segmentation", "detect_model_type"),
    "load_sam_model": (".sam_segmentation", "load_sam_model"),
    "generate_masks": (".sam_segmentation", "generate_masks"),
    "convert_masks_to_segments": (".sam_segmentation", "convert_masks_to_segments"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    return resolve_attribute(globals(), __name__, name, _LAZY_IMPORTS)


def __dir__():
    return public_dir(globals(), _LAZY_IMPORTS)
