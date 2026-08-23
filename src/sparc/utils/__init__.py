"""SPARC array, geometry, threading, Pancam, and SEL helpers."""

from .._lazy import public_dir, resolve_attribute


_LAZY_IMPORTS = {
    "mask_cube": (".array_ops", "mask_cube"),
    "uncompress_cube": (".array_ops", "uncompress_cube"),
    "apply_kmeans_to_masked": (".array_ops", "apply_kmeans_to_masked"),
    "normalize_cube": (".array_ops", "normalize_cube"),
    "normalize_minmax": (".array_ops", "normalize_minmax"),
    "normalize_zscore": (".array_ops", "normalize_zscore"),
    "normalize_l2": (".array_ops", "normalize_l2"),
    "compute_spectral_statistics": (".array_ops", "compute_spectral_statistics"),
    "find_center_of_mass": (".geometry", "find_center_of_mass"),
    "find_largest_rectangle": (".geometry", "find_largest_rectangle"),
    "check_rectangle_valid": (".geometry", "check_rectangle_valid"),
    "extract_roi": (".geometry", "extract_roi"),
    "create_edge_mask": (".geometry", "create_edge_mask"),
    "convert_to_plot_coords": (".geometry", "convert_to_plot_coords"),
    "create_rgb_image": (".geometry", "create_rgb_image"),
    "get_edge_mask": (".geometry", "get_edge_mask"),
    "get_roi": (".geometry", "get_roi"),
    "rect_to_plot_coords": (".geometry", "rect_to_plot_coords"),
    "get_rgb_stretch": (".geometry", "get_rgb_stretch"),
    "SafeKMeans": (".threading", "SafeKMeans"),
    "parse_pcam_fn": (".pancam_helpers", "parse_pcam_fn"),
    "scan_pcam_files": (".pancam_helpers", "scan_pcam_files"),
    "get_pcam_bandset": (".pancam_helpers", "get_pcam_bandset"),
    "export_sel": (".sel_writer", "export_sel"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    return resolve_attribute(globals(), __name__, name, _LAZY_IMPORTS)


def __dir__():
    return public_dir(globals(), _LAZY_IMPORTS)
