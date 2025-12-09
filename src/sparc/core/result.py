"""Result handling and visualization for SPARC pipeline."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from .state import SparcState
from ..visualization.plotting import plot_roi_image, plot_spectra


@dataclass
class SparcResult:
    """Immutable result from SPARC pipeline."""
    scene_id: str
    final_rois: np.ndarray
    final_spectra: np.ndarray
    final_stds: np.ndarray
    n_segments: int
    n_clusters: int
    rgb_img: np.ndarray
    segments: np.ndarray
    clustering_result: Dict[str, Any]
    
    @classmethod
    def from_state(cls, state: SparcState) -> 'SparcResult':
        """
        Create result from pipeline state.
        
        Args:
            state: Completed pipeline state
            
        Returns:
            SparcResult instance
        """
        return cls(
            scene_id=state.load_result['id'],
            final_rois=state.final_rois,
            final_spectra=state.final_spectra,
            final_stds=state.final_stds,
            n_segments=len(np.unique(state.segments)),
            n_clusters=state.clustering_result['n_components'],
            rgb_img=state.load_result['rgb_img'],
            segments=state.segments,
            clustering_result=state.clustering_result
        )


def plot_result(result: SparcResult,
               show_segments: bool = True,
               show_rois: bool = True,
               show_spectra: bool = True,
               figsize: tuple = (15, 10)) -> plt.Figure:
    """
    Plot SPARC pipeline results.
    
    Args:
        result: SparcResult to visualize
        show_segments: Show segmentation
        show_rois: Show final ROIs
        show_spectra: Show final spectra
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    plots = []
    if show_segments:
        plots.append('segments')
    if show_rois:
        plots.append('rois')
    if show_spectra:
        plots.append('spectra')
    
    if not plots:
        raise ValueError("No plots selected")
    
    fig, axes = plt.subplots(1, len(plots), figsize=figsize)
    if len(plots) == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if 'segments' in plots:
        axes[plot_idx].imshow(result.segments, cmap='tab20')
        axes[plot_idx].set_title(f'Segmentation ({result.n_segments} segments)')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    if 'rois' in plots:
        plot_roi_image(
            result.rgb_img,
            result.final_rois,
            ax=axes[plot_idx],
            show=False
        )
        axes[plot_idx].set_title(f'Final ROIs ({len(result.final_rois)})')
        plot_idx += 1
    
    if 'spectra' in plots:
        plot_spectra(
            result.final_spectra,
            result.final_stds,
            ax=axes[plot_idx],
            show=False
        )
        axes[plot_idx].set_title(f'Final Spectra ({len(result.final_spectra)} clusters)')
        plot_idx += 1
    
    plt.tight_layout()
    return fig


def export_spectra_csv(result: SparcResult, output_path: str):
    """
    Export final spectra to CSV.
    
    Args:
        result: SparcResult containing spectra
        output_path: Path for CSV output
    """
    import pandas as pd
    from ..core.constants import WLS
    
    data = {
        'ROI': np.repeat(np.arange(len(result.final_spectra)), len(WLS)),
        'Wavelength': np.tile(WLS, len(result.final_spectra)),
        'Reflectance': result.final_spectra.flatten(),
        'StdDev': result.final_stds.flatten()
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def export_rois_json(result: SparcResult, output_path: str):
    """
    Export ROI coordinates to JSON.
    
    Args:
        result: SparcResult containing ROIs
        output_path: Path for JSON output
    """
    import json
    
    rois = [
        {
            'id': i,
            'x': int(roi[0]),
            'y': int(roi[1]),
            'width': int(roi[2]),
            'height': int(roi[3]),
            'cluster': int(result.clustering_result['labels'][i])
        }
        for i, roi in enumerate(result.final_rois)
    ]
    
    data = {
        'scene_id': result.scene_id,
        'n_rois': len(rois),
        'n_clusters': result.n_clusters,
        'rois': rois
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
