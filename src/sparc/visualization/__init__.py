"""SPARC visualization module - Plotting and result visualization."""

from .plotting import (
    plot_rois_on_image,
    plot_spectra_with_error,
    plot_clustering,
    create_summary_plot,
    # Backward compatibility aliases
    plot_roi_image,
    plot_spectra,
    plot_clustering_results,
    plot_pipeline_summary,
)

__all__ = [
    'plot_rois_on_image',
    'plot_spectra_with_error',
    'plot_clustering',
    'create_summary_plot',
    'plot_roi_image',  # Alias
    'plot_spectra',  # Alias
    'plot_clustering_results',  # Alias
    'plot_pipeline_summary',  # Alias
]