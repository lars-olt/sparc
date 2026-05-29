"""Visualization and plotting functionality."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List

from ..core.constants import WAVELENGTHS, COLORS, PLOT_MARKERS

def plot_rois_on_image(img: np.ndarray,
                       rois: np.ndarray,
                       ax: Optional[plt.Axes] = None,
                       show: bool = True) -> plt.Figure:
    """
    Plot ROI rectangles overlaid on image.
    
    Args:
        img: RGB image to display
        rois: Array of ROI coordinates (x, y, width, height)
        ax: Optional axes to plot on
        show: Whether to display plot
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.frameon = False
    else:
        fig = ax.get_figure()
    
    ax.set_axis_off()
    ax.imshow(img)
    
    for i, (x, y, width, height) in enumerate(rois):
        color = COLORS[i % len(COLORS)]
        
        rectangle = patches.Rectangle(
            (x, y), width, height,
            edgecolor=color,
            facecolor="none",
            linewidth=2
        )
        ax.add_patch(rectangle)
    
    if show:
        plt.show()
    
    return fig


def plot_spectra_with_error(spectra: np.ndarray,
                            stds: np.ndarray,
                            ax: Optional[plt.Axes] = None,
                            colors: List[str] = COLORS,
                            show: bool = True,
                            wavelengths: Optional[List[float]] = None) -> plt.Figure:
    """
    Plot spectra with error bars.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    wls = np.array(wavelengths if wavelengths is not None else WAVELENGTHS)
    n_bands = spectra.shape[1]
    wls = wls[:n_bands]  # guard against any length mismatch

    sorted_indices = np.argsort(wls)
    sorted_wls = wls[sorted_indices]

    for i, spectrum in enumerate(spectra):
        color = colors[i % len(colors)]
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        ax.errorbar(
            sorted_wls,
            spectrum[sorted_indices],
            yerr=stds[i][sorted_indices],
            fmt="-",
            marker=marker,
            ecolor=color,
            capsize=3,
            color=color
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("IOF")
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig


def plot_clustering(spectra: np.ndarray,
                   labels: np.ndarray,
                   title: str = "Spectral Clustering",
                   show: bool = True) -> plt.Figure:
    """
    Plot spectra colored by cluster assignment.
    
    Args:
        spectra: Array of spectra
        labels: Cluster labels
        title: Plot title
        show: Whether to display plot
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wavelengths = np.array(WAVELENGTHS[3:])
    sorted_indices = np.argsort(wavelengths)
    sorted_wls = wavelengths[sorted_indices]
    
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        cluster_spectra = spectra[mask]
        color = COLORS[i % len(COLORS)]
        
        for spectrum in cluster_spectra:
            sorted_spectrum = spectrum[sorted_indices]
            ax.plot(sorted_wls, sorted_spectrum, color=color, alpha=0.7)
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Reflectance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig


def create_summary_plot(result_dict: dict, show: bool = True) -> plt.Figure:
    """
    Create summary plot showing complete pipeline results.
    
    Args:
        result_dict: Dictionary containing pipeline results
        show: Whether to display plot
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(result_dict['rgb_img'])
    axes[0, 0].set_title("Original RGB Image")
    axes[0, 0].set_axis_off()
    
    # Segmentation
    axes[0, 1].imshow(result_dict['segments'], cmap='tab20')
    axes[0, 1].set_title(f"Segmentation ({result_dict['n_segments']} segments)")
    axes[0, 1].set_axis_off()
    
    # Final ROIs
    plot_rois_on_image(
        result_dict['rgb_img'],
        result_dict['final_rois'],
        ax=axes[1, 0],
        show=False
    )
    axes[1, 0].set_title(f"Selected ROIs ({len(result_dict['final_rois'])})")
    
    # Final spectra
    if 'final_spectra' in result_dict and 'final_stds' in result_dict:
        plot_spectra_with_error(
            result_dict['final_spectra'],
            result_dict['final_stds'],
            ax=axes[1, 1],
            show=False
        )
        axes[1, 1].set_title(f"Final Spectra ({len(result_dict['final_spectra'])} clusters)")
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


# Aliases for backward compatibility
plot_roi_image = plot_rois_on_image
plot_spectra = plot_spectra_with_error
plot_clustering_results = plot_clustering
plot_pipeline_summary = create_summary_plot