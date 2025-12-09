"""
SPARC: Spectral Analysis and ROI Classification pipeline.
"""

from .core.sparc import Sparc
from .core.functional import run_sparc, run_sparc_steps
from .core.config import (
    SparcConfig,
    LoadConfig,
    PreprocessConfig,
    SegmentConfig,
    ROIConfig,
    SpectralConfig,
    PerformanceConfig,
    SegmentationBackend,
    ROIBackend
)
from .core.state import SparcState
from .core.result import (
    SparcResult, 
    plot_result, 
    export_spectra_csv, 
    export_rois_json
)
from .core.logging_utils import setup_logger, configure_logging

__all__ = [
    "Sparc",
    "run_sparc",
    "run_sparc_steps",
    "SparcConfig",
    "LoadConfig",
    "PreprocessConfig",
    "SegmentConfig",
    "ROIConfig",
    "SpectralConfig",
    "PerformanceConfig",
    "SegmentationBackend",
    "ROIBackend",
    "SparcState",
    "SparcResult",
    "plot_result",
    "export_spectra_csv",
    "export_rois_json",
    "setup_logger",
    "configure_logging",
]