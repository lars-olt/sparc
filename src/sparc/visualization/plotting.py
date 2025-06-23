"""Visualization and plotting functionality."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List

from ..core.constants import WLS, COLORS, MARKERS


def plot_roi_image(img: np.ndarray, 
                  rois: np.ndarray, 
                  ax: Optional[plt.Axes] = None,
                  show: bool = True) -> plt.Figure:
    """
    Plot ROI rectangles overlaid on an image.
    
    Args:
        img: RGB image to display
        rois: Array of ROI coordinates in (x, y, width, height) format
        ax: Optional matplotlib axes to plot on
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.frameon = False
    else:
        fig = ax.get_figure()
    
    ax.set_axis_off()
    ax.imshow(img)

    color_i = 0
    for (x, y, w, h) in rois:
        # Cycle through colors
        if color_i == len(COLORS):
            color_i = 0
        curr_color = COLORS[color_i]
        color_i += 1

        # Add ROI rectangle (original expects x, y, width, height format)
        roi = patches.Rectangle(
            (x, y), w, h, 
            edgecolor=curr_color, 
            facecolor="none", 
            linewidth=2
        )
        ax.add_patch(roi)

    if show:
        plt.show()
        
    return fig


def plot_spectra(spectra: np.ndarray, 
                stds: np.ndarray, 
                ax: Optional[plt.Axes] = None,
                colors: List[str] = COLORS,
                show: bool = True) -> plt.Figure:
    """
    Plot averaged spectra with error bars.
    
    Args:
        spectra: Array of spectra to plot
        stds: Standard deviations for error bars
        ax: Optional matplotlib axes to plot on
        colors: List of colors to cycle through
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # Sort wavelengths for proper plotting
    bayer_sorted_indices = np.argsort(WLS[:3])
    non_bayer_sorted_indices = np.argsort(WLS[3:]) + 3

    color_i = 0
    marker_i = 0
    
    for i, spectrum in enumerate(spectra):
        # Cycle colors if needed
        if color_i == len(colors):
            color_i = 0

        # Cycle markers if needed
        if marker_i == len(MARKERS):
            marker_i = 0

        curr_color = colors[color_i]

        # Plot non-Bayer bands with error bars
        nb_wls = np.array(WLS)[non_bayer_sorted_indices]
        nb_data = spectrum[non_bayer_sorted_indices]
        ax.errorbar(
            nb_wls,
            nb_data,
            yerr=stds[i][non_bayer_sorted_indices],
            fmt="-",
            ecolor=curr_color,
            capsize=3,
            color=curr_color
        )

        # Plot Bayer bands as points
        b_wls = np.array(WLS)[bayer_sorted_indices]
        b_data = spectrum[bayer_sorted_indices]
        ax.plot(b_wls, b_data, "+", color=curr_color)

        color_i += 1
        marker_i += 1

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("R* = IOF/cos(θ)")
    
    if show:
        plt.show()
        
    return fig


def plot_clustering_results(spectra: np.ndarray,
                          labels: np.ndarray,
                          title: str = "Spectral Clustering Results",
                          show: bool = True) -> plt.Figure:
    """
    Plot spectra colored by cluster assignment.
    
    Args:
        spectra: Array of spectra
        labels: Cluster labels for each spectrum
        title: Plot title
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use wavelengths for x-axis
    wavelengths = np.array(WLS[3:])  # Non-Bayer wavelengths
    sorted_indices = np.argsort(wavelengths)
    sorted_wls = wavelengths[sorted_indices]
    
    # Plot each spectrum colored by cluster
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
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


def plot_pipeline_summary(sparc_instance, show: bool = True) -> plt.Figure:
    """
    Create a summary plot showing the complete SPARC pipeline results.
    
    Args:
        sparc_instance: SPARC class instance with completed pipeline
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(sparc_instance.load_result['rgb_img'])
    axes[0, 0].set_title("Original RGB Image")
    axes[0, 0].set_axis_off()
    
    # Segmentation
    axes[0, 1].imshow(sparc_instance.segments, cmap='tab20')
    axes[0, 1].set_title("SAM Segmentation")
    axes[0, 1].set_axis_off()
    
    # Final ROIs
    plot_roi_image(
        sparc_instance.load_result['rgb_img'], 
        sparc_instance.final_rois, 
        ax=axes[1, 0],
        show=False
    )
    axes[1, 0].set_title("Selected ROIs")
    
    # Final spectra
    if sparc_instance.roi_indices is not None:
        if len(sparc_instance.roi_spectra[sparc_instance.outlier_mask]) > 3:
            outlier_spectra = sparc_instance.roi_spectra[sparc_instance.outlier_mask]
            outlier_stds = sparc_instance.roi_stds[sparc_instance.outlier_mask]
            final_spectra = outlier_spectra[sparc_instance.roi_indices]
            final_stds = outlier_stds[sparc_instance.roi_indices]
        else:
            final_spectra = sparc_instance.roi_spectra[sparc_instance.roi_indices]
            final_stds = sparc_instance.roi_stds[sparc_instance.roi_indices]
        
        plot_spectra(final_spectra, final_stds, ax=axes[1, 1], show=False)
        axes[1, 1].set_title("Final Spectra")
    
    plt.tight_layout()
    
    if show:
        plt.show()
        
    return fig