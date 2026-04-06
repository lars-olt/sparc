"""SPARC core module - Pipeline orchestration and configuration."""

from .sparc import Sparc
from .functional import run_sparc, run_sparc_steps
from .config import (
    SparcConfig,
    LoadConfig,
    PreprocessConfig,
    SegmentConfig,
    ROIConfig,
    SpectralConfig,
    PerformanceConfig,
    SegmentationBackend,
    ROIBackend,
)
from .state import SparcState
from .result import SparcResult, plot_result, export_spectra_csv, export_rois_json, export_sel
from .logging_utils import setup_logger
from .constants import (
    WAVELENGTHS,
    BAYER_CUTOFF_INDEX,
    LEFT_CUTOFF_INDEX,
    RGB_BANDS,
    COLOR_MAPPINGS,
    COLOR_NAMES,
    COLORS,
    PLOT_MARKERS,
    SHARED_BANDS,
    BAD_PIXEL_FLAGS,
    DEFAULT_ROI_AREA_THRESHOLD,
    DEFAULT_EDGE_OFFSET,
    DEFAULT_ALLOWED_VARIANCE,
    DEFAULT_ALBEDO_RATIO_THRESHOLD,
)

__all__ = [
    # Main classes
    'Sparc',
    'run_sparc',
    'run_sparc_steps',
    
    # Configuration
    'SparcConfig',
    'LoadConfig',
    'PreprocessConfig',
    'SegmentConfig',
    'ROIConfig',
    'SpectralConfig',
    'PerformanceConfig',
    'SegmentationBackend',
    'ROIBackend',
    
    # State and results
    'SparcState',
    'SparcResult',
    'plot_result',
    'export_spectra_csv',
    'export_rois_json',
    'export_sel',
    
    # Utilities
    'setup_logger',
    
    # Constants
    'WAVELENGTHS',
    'BAYER_CUTOFF_INDEX',
    'LEFT_CUTOFF_INDEX',
    'RGB_BANDS',
    'COLOR_MAPPINGS',
    'COLOR_NAMES',
    'COLORS',
    'PLOT_MARKERS',
    'SHARED_BANDS',
    'BAD_PIXEL_FLAGS',
    'DEFAULT_ROI_AREA_THRESHOLD',
    'DEFAULT_EDGE_OFFSET',
    'DEFAULT_ALLOWED_VARIANCE',
    'DEFAULT_ALBEDO_RATIO_THRESHOLD',
]