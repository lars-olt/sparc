"""SPARC core orchestration and configuration, exposed lazily."""

from .._lazy import public_dir, resolve_attribute


_LAZY_IMPORTS = {
    "Sparc": (".sparc", "Sparc"),
    "run_sparc": (".functional", "run_sparc"),
    "run_sparc_steps": (".functional", "run_sparc_steps"),
    "SparcConfig": (".config", "SparcConfig"),
    "LoadConfig": (".config", "LoadConfig"),
    "PreprocessConfig": (".config", "PreprocessConfig"),
    "SegmentConfig": (".config", "SegmentConfig"),
    "ROIConfig": (".config", "ROIConfig"),
    "SpectralConfig": (".config", "SpectralConfig"),
    "PerformanceConfig": (".config", "PerformanceConfig"),
    "SegmentationBackend": (".config", "SegmentationBackend"),
    "ROIBackend": (".config", "ROIBackend"),
    "SparcState": (".state", "SparcState"),
    "SparcResult": (".result", "SparcResult"),
    "plot_result": (".result", "plot_result"),
    "export_spectra_csv": (".result", "export_spectra_csv"),
    "export_rois_json": (".result", "export_rois_json"),
    "export_sel": (".result", "export_sel"),
    "setup_logger": (".logging_utils", "setup_logger"),
    "WAVELENGTHS": (".constants", "WAVELENGTHS"),
    "BAYER_CUTOFF_INDEX": (".constants", "BAYER_CUTOFF_INDEX"),
    "LEFT_CUTOFF_INDEX": (".constants", "LEFT_CUTOFF_INDEX"),
    "RGB_BANDS": (".constants", "RGB_BANDS"),
    "COLOR_MAPPINGS": (".constants", "COLOR_MAPPINGS"),
    "COLOR_NAMES": (".constants", "COLOR_NAMES"),
    "COLORS": (".constants", "COLORS"),
    "PLOT_MARKERS": (".constants", "PLOT_MARKERS"),
    "SHARED_BANDS": (".constants", "SHARED_BANDS"),
    "BAD_PIXEL_FLAGS": (".constants", "BAD_PIXEL_FLAGS"),
    "DEFAULT_ROI_AREA_THRESHOLD": (".constants", "DEFAULT_ROI_AREA_THRESHOLD"),
    "DEFAULT_EDGE_OFFSET": (".constants", "DEFAULT_EDGE_OFFSET"),
    "DEFAULT_ALLOWED_VARIANCE": (".constants", "DEFAULT_ALLOWED_VARIANCE"),
    "DEFAULT_ALBEDO_RATIO_THRESHOLD": (
        ".constants",
        "DEFAULT_ALBEDO_RATIO_THRESHOLD",
    ),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    return resolve_attribute(globals(), __name__, name, _LAZY_IMPORTS)


def __dir__():
    return public_dir(globals(), _LAZY_IMPORTS)
