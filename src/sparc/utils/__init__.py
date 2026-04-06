"""SPARC utilities module - Array operations, geometry, threading, and helpers."""

from .array_ops import (
    mask_cube,
    uncompress_cube,
    apply_kmeans_to_masked,
    normalize_cube,
    normalize_minmax,
    normalize_zscore,
    normalize_l2,
    compute_spectral_statistics,
)
from .geometry import (
    find_center_of_mass,
    find_largest_rectangle,
    check_rectangle_valid,
    extract_roi,
    create_edge_mask,
    convert_to_plot_coords,
    create_rgb_image,
    get_edge_mask,
    get_roi,
    rect_to_plot_coords,
    get_rgb_stretch,
)
from .threading import (
    SafeKMeans,
)
from .pancam_helpers import (
    parse_pcam_fn,
    scan_pcam_files,
    get_pcam_bandset,
)

from .sel_writer import export_sel

__all__ = [
    # Array operations
    'mask_cube',
    'uncompress_cube',
    'apply_kmeans_to_masked',
    'normalize_cube',
    'normalize_minmax',
    'normalize_zscore',
    'normalize_l2',
    'compute_spectral_statistics',

    # Geometry
    'find_center_of_mass',
    'find_largest_rectangle',
    'check_rectangle_valid',
    'extract_roi',
    'create_edge_mask',
    'convert_to_plot_coords',
    'create_rgb_image',
    'get_edge_mask',
    'get_roi',
    'rect_to_plot_coords',
    'get_rgb_stretch',

    # Threading
    'SafeKMeans',

    # Pancam helpers
    'parse_pcam_fn',
    'scan_pcam_files',
    'get_pcam_bandset',
    
    'export_sel'
]