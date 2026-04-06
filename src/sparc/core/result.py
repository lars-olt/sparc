"""Result handling and visualization for SPARC pipeline."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from .state import SparcState
from ..visualization.plotting import plot_roi_image, plot_spectra


@dataclass
class SparcResult:
    """Immutable result from SPARC pipeline."""
    scene_id: str
    instrument: str
    final_rois: np.ndarray        # right-camera space (x, y, w, h), cropped coords
    final_left_rois: np.ndarray   # left-camera inscribed rects (x, y, w, h), cropped coords
    final_spectra: np.ndarray
    final_stds: np.ndarray
    n_segments: int
    n_clusters: int
    rgb_img: np.ndarray
    segments: np.ndarray
    clustering_result: Dict[str, Any]
    wavelengths: List[float]

    _load_result: Optional[Dict[str, Any]] = None

    @classmethod
    def from_state(cls, state: SparcState) -> 'SparcResult':
        instrument_cfg  = getattr(state, 'instrument_config', None)
        instrument      = instrument_cfg['instrument'] if instrument_cfg else 'ZCAM'
        all_wavelengths = instrument_cfg['wavelengths'] if instrument_cfg else []
        n_bands         = state.final_spectra.shape[1]
        wavelengths     = all_wavelengths[:n_bands]

        return cls(
            scene_id          = state.load_result['id'],
            instrument        = instrument,
            final_rois        = state.final_rois,
            final_left_rois   = state.final_left_rois,
            final_spectra     = state.final_spectra,
            final_stds        = state.final_stds,
            n_segments        = len(np.unique(state.segments)),
            n_clusters        = state.clustering_result['n_components'],
            rgb_img           = state.load_result['rgb_img'],
            segments          = state.segments,
            clustering_result = state.clustering_result,
            wavelengths       = wavelengths,
            _load_result      = state.load_result,
        )


def plot_result(result: SparcResult,
               show_segments: bool = True,
               show_rois: bool = True,
               show_spectra: bool = True,
               figsize: tuple = (15, 10)) -> plt.Figure:
    plots = [p for p, show in [('segments', show_segments), ('rois', show_rois), ('spectra', show_spectra)] if show]
    if not plots:
        raise ValueError("At least one plot type must be enabled.")

    n_plots    = len(plots)
    panel_size = figsize[0] / n_plots
    fig, axes  = plt.subplots(1, n_plots, figsize=(figsize[0], panel_size))
    if n_plots == 1:
        axes = [axes]

    for idx, plot_type in enumerate(plots):
        if plot_type == 'segments':
            axes[idx].imshow(result.segments, cmap='tab20')
            axes[idx].set_title(f'Segmentation ({result.n_segments} segments)')
            axes[idx].axis('off')
        elif plot_type == 'rois':
            plot_roi_image(result.rgb_img, result.final_rois, ax=axes[idx], show=False)
            axes[idx].set_title(f'Final ROIs ({len(result.final_rois)}) [{result.instrument}]')
        elif plot_type == 'spectra':
            axes[idx].set_aspect('auto')
            plot_spectra(result.final_spectra, result.final_stds,
                         ax=axes[idx], show=False, wavelengths=result.wavelengths)
            axes[idx].set_title(f'Final Spectra ({len(result.final_spectra)} clusters)')

    plt.tight_layout()
    plt.close(fig)
    return fig


def export_spectra_csv(result: SparcResult, output_path: str):
    import pandas as pd
    data = {
        'ROI':         np.repeat(np.arange(len(result.final_spectra)), len(result.wavelengths)),
        'Wavelength':  np.tile(result.wavelengths, len(result.final_spectra)),
        'Reflectance': result.final_spectra[:, :len(result.wavelengths)].flatten(),
        'StdDev':      result.final_stds[:, :len(result.wavelengths)].flatten(),
    }
    pd.DataFrame(data).to_csv(output_path, index=False)


def export_rois_json(result: SparcResult, output_path: str):
    import json
    rois = [
        {
            'id':      i,
            'x':       int(roi[0]), 'y': int(roi[1]),
            'width':   int(roi[2]), 'height': int(roi[3]),
            'cluster': int(result.clustering_result['labels'][i]),
        }
        for i, roi in enumerate(result.final_rois)
    ]
    data = {
        'scene_id':   result.scene_id,
        'instrument': result.instrument,
        'n_rois':     len(rois),
        'n_clusters': result.n_clusters,
        'rois':       rois,
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def export_sel(result: SparcResult, output_path: str, template_path: Optional[str] = None) -> None:
    """Export ROIs as a merspect-compatible .sel file.

    ZCAM ROIs are stored internally in cropped-image space (post ZCAM_CROP) and
    are shifted back to full-sensor coordinates before writing. Pancam ROIs are
    already in full sensor space.

    Args:
        result:        Completed SparcResult.
        output_path:   Destination path, e.g. "scene.sel".
        template_path: Optional path to a blank .sel template. Defaults to the
                       packaged blank for the result's instrument.
    """
    from ..utils.sel_writer import export_sel as _write_sel, filenames_from_load_result

    n_rois      = len(result.final_rois)
    load_result = result._load_result
    instrument  = str(result.instrument).strip().upper()

    if instrument in {"ZCAM", "MCZ"} and load_result is not None:
        # ZCAM_CROP = (left, right, top, bottom). The pipeline crops the raw
        # 1648x1200 frame before processing, so we add the offsets back here.
        from asdf_settings import rapidlooks
        crop     = rapidlooks.CROP_SETTINGS["crop"]
        col_off  = crop[0]
        row_off  = crop[2]

        raw_band       = next(iter(load_result["base_bands"].values()))
        full_H, full_W = raw_band.shape
    else:
        col_off, row_off = 0, 0
        full_H, full_W   = result.rgb_img.shape[:2]

    def _shift(rois: np.ndarray) -> np.ndarray:
        if rois.size == 0:
            return np.zeros((0, 4), dtype=np.int32)
        shifted      = rois.copy().astype(np.int32)
        shifted[:, 0] += col_off
        shifted[:, 1] += row_off
        return shifted

    left_names, right_names = (
        filenames_from_load_result(load_result, n_rois)
        if load_result is not None
        else ([result.scene_id] * n_rois, [result.scene_id] * n_rois)
    )

    _write_sel(
        output_path     = output_path,
        final_rois      = _shift(result.final_rois),
        final_left_rois = _shift(result.final_left_rois),
        image_shape     = (full_H, full_W),
        left_filenames  = left_names,
        right_filenames = right_names,
        template_path   = template_path,
        instrument      = result.instrument,
    )