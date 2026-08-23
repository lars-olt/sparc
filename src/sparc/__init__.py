"""SPARC: Spectral Analysis and ROI Classification pipeline.

The public API is resolved lazily so importing the package for its loaders and
file helpers does not import the optional segmentation or clustering stack.
"""

from ._lazy import public_dir, resolve_attribute


_LAZY_IMPORTS = {
    "Sparc": (".core.sparc", "Sparc"),
    "run_sparc": (".core.functional", "run_sparc"),
    "run_sparc_steps": (".core.functional", "run_sparc_steps"),
    "SparcConfig": (".core.config", "SparcConfig"),
    "LoadConfig": (".core.config", "LoadConfig"),
    "PreprocessConfig": (".core.config", "PreprocessConfig"),
    "SegmentConfig": (".core.config", "SegmentConfig"),
    "ROIConfig": (".core.config", "ROIConfig"),
    "SpectralConfig": (".core.config", "SpectralConfig"),
    "PerformanceConfig": (".core.config", "PerformanceConfig"),
    "SegmentationBackend": (".core.config", "SegmentationBackend"),
    "ROIBackend": (".core.config", "ROIBackend"),
    "SparcState": (".core.state", "SparcState"),
    "SparcResult": (".core.result", "SparcResult"),
    "plot_result": (".core.result", "plot_result"),
    "export_spectra_csv": (".core.result", "export_spectra_csv"),
    "export_rois_json": (".core.result", "export_rois_json"),
    "setup_logger": (".core.logging_utils", "setup_logger"),
    "configure_logging": (".core.logging_utils", "configure_logging"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    return resolve_attribute(globals(), __name__, name, _LAZY_IMPORTS)


def __dir__():
    return public_dir(globals(), _LAZY_IMPORTS)
