"""Main SPARC class for Spectral Pattern Analysis for ROI Classification."""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path

from ..data.loading import load_cube, LoadResult
from ..preprocessing.masking import apply_masking
from ..preprocessing.calibration import apply_photometric_calibration
from ..segmentation.sam_segmentation import segment_image
from ..roi.extraction import get_potential_rois
from ..roi.filtering import (
    filter_rois,
    apply_selection_heuristics,
    filter_by_albedo_ratio,
)
from ..spectral.analysis import detect_unique_spectra, auto_bayesian_gmm
from ..spectral.metrics import average_spectra
from ..visualization.plotting import plot_roi_image, plot_spectra
from ..utils.geometry import rect_to_plot_coords
from .constants import *


class Sparc:
    """
    Spectral Pattern Analysis for ROI Classification.

    This class provides a complete pipeline for analyzing hyperspectral images
    to identify and classify regions of interest based on their spectral signatures.
    """

    def __init__(self, sam_model_path: str):
        """
        Initialize SPARC with SAM model path.

        Args:
            sam_model_path: Path to the SAM model weights file
        """
        self.sam_model_path = sam_model_path
        self.reset()

    def reset(self):
        """Reset all stored results and state."""
        # Raw data
        self.load_result: Optional[LoadResult] = None
        self.using_pixmaps: bool = False

        # Preprocessing results
        self.processed_data: Optional[np.ndarray] = None
        self.photometrically_calibrated: Optional[np.ndarray] = None
        self.shadow_mask: Optional[np.ndarray] = None
        self.sky_mask: Optional[np.ndarray] = None
        self.full_mask: Optional[np.ndarray] = None

        # Segmentation results
        self.segments: Optional[np.ndarray] = None

        # ROI results
        self.unfiltered_rois: Optional[np.ndarray] = None
        self.area_filtered_rois: Optional[np.ndarray] = None
        self.albedo_valid_indices: Optional[np.ndarray] = None
        self.final_rois: Optional[np.ndarray] = None
        self.roi_indices: Optional[List[int]] = None

        # Spectral analysis results
        self.roi_spectra: Optional[np.ndarray] = None
        self.roi_stds: Optional[np.ndarray] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self.clustering_result: Optional[Dict[str, Any]] = None
        self.all_clustering_results: Optional[Dict[str, Any]] = None

        # Configuration
        self.config: Dict[str, Any] = {}

    def load_data(
        self,
        iof_path: str,
        seq_id: Optional[str] = None,
        obs_ix: int = 0,
        do_apply_pixmaps: bool = True,
        ignore_bayers: bool = False,
    ) -> "Sparc":
        """
        Load hyperspectral data cube.

        Args:
            iof_path: Path to IOF data
            seq_id: Sequence ID (optional)
            obs_ix: Observation index
            do_apply_pixmaps: Whether to apply pixel maps
            ignore_bayers: Whether to ignore Bayer filters

        Returns:
            Self for method chaining
        """
        load_kwargs = {
            "iof_path": iof_path,
            "seq_id": seq_id,
            "obs_ix": obs_ix,
            "do_apply_pixmaps": do_apply_pixmaps,
            "ignore_bayers": ignore_bayers,
        }

        self.load_result = load_cube(**load_kwargs)
        self.using_pixmaps = do_apply_pixmaps
        self.config["load_kwargs"] = load_kwargs

        return self

    def preprocess(
        self,
        shadow_kwargs: Optional[Dict] = None,
        skymask_kwargs: Optional[Dict] = None,
        apply_r_star: bool = True,
    ) -> "Sparc":
        """
        Preprocess the data with masking and calibration.

        Args:
            shadow_kwargs: Parameters for shadow masking
            skymask_kwargs: Parameters for sky masking
            apply_r_star: Whether to apply photometric calibration

        Returns:
            Self for method chaining
        """
        if self.load_result is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Default parameters
        if shadow_kwargs is None:
            shadow_kwargs = {"percentiles": (20, 100), "operator": "and"}
        if skymask_kwargs is None:
            skymask_kwargs = {
                "percentile": 75,
                "edge_params": {"maximum": 5, "erosion": 3},
                "input_median": 5,
                "trace_maximum": 5,
                "cutoffs": {"extent": 0.05, "coverage": None, "v": 0.9, "h": None},
                "input_mask_dilation": None,
                "input_stretch": (10, 1),
                "floodfill": True,
                "trim_params": {"trim": False},
                "clear": True,
                "colorblock": False,
                "respect_mask": False,
            }

        # Apply masking
        mask_result = apply_masking(
            self.load_result, self.using_pixmaps, shadow_kwargs, skymask_kwargs
        )

        self.processed_data = mask_result["masked_cube"]
        self.shadow_mask = mask_result["shadow_mask"]
        self.sky_mask = mask_result["sky_mask"]
        self.full_mask = mask_result["full_mask"]

        # Apply photometric calibration
        self.photometrically_calibrated = apply_photometric_calibration(
            self.processed_data, self.load_result, apply_r_star
        )

        self.config["preprocess_kwargs"] = {
            "shadow_kwargs": shadow_kwargs,
            "skymask_kwargs": skymask_kwargs,
            "apply_r_star": apply_r_star,
        }

        return self

    def segment(self) -> "Sparc":
        """
        Segment the RGB image using SAM.

        Returns:
            Self for method chaining
        """
        if self.load_result is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.segments = segment_image(self.sam_model_path, self.load_result["rgb_img"])
        return self

    def extract_rois(
        self,
        edge_offset: int = DEFAULT_EDGE_OFFSET,
        allowed_variance: float = DEFAULT_ALLOWED_VARIANCE,
    ) -> "Sparc":
        """
        Extract potential regions of interest.

        Args:
            edge_offset: Offset from image edges
            allowed_variance: Maximum variance allowed in clustering

        Returns:
            Self for method chaining
        """
        if self.segments is None:
            raise ValueError("No segmentation available. Call segment() first.")
        if self.processed_data is None:
            raise ValueError("No preprocessed data. Call preprocess() first.")

        self.unfiltered_rois = get_potential_rois(
            self.segments, self.processed_data, edge_offset, allowed_variance
        )

        self.config["roi_kwargs"] = {
            "edge_offset": edge_offset,
            "allowed_variance": allowed_variance,
        }

        return self

    def filter_rois(self, area_threshold: int = DEFAULT_ROI_AREA_THRESHOLD) -> "Sparc":
        """
        Filter ROIs by area and compute spectra.

        Args:
            area_threshold: Minimum area threshold for ROIs

        Returns:
            Self for method chaining
        """
        if self.unfiltered_rois is None:
            raise ValueError("No ROIs extracted. Call extract_rois() first.")
        if self.photometrically_calibrated is None:
            raise ValueError("No calibrated data. Call preprocess() first.")

        # Area-filtered ROIs and their indices
        self.area_filtered_rois = filter_rois(self.unfiltered_rois, area_threshold)

        if len(self.area_filtered_rois) == 0:
            raise ValueError(
                f"No ROIs remaining after area filtering with threshold {area_threshold}"
            )

        # Convert to plot coordinates and compute spectra
        plt_rois = rect_to_plot_coords(self.area_filtered_rois)
        self.roi_spectra, self.roi_stds = average_spectra(
            self.photometrically_calibrated, plt_rois
        )

        # Apply albedo ratio filtering as in original code
        # NOTE: Original code has a bug - it filters spectra but not ROIs
        # We replicate this behavior for consistency
        filtered_spectra, filtered_stds, albedo_valid_indices = filter_by_albedo_ratio(
            self.roi_spectra, self.roi_stds, DEFAULT_ALBEDO_RATIO_THRESHOLD
        )

        # Store the albedo filtering indices to track correspondence
        self.albedo_valid_indices = albedo_valid_indices

        # Update spectra but keep original ROIs (matches original behavior)
        self.roi_spectra = filtered_spectra
        self.roi_stds = filtered_stds
        # Keep self.area_filtered_rois unchanged to match original bug/behavior

        self.config["area_threshold"] = area_threshold
        return self

    def analyze_spectra(
        self,
        contamination: float = 0.1,
        freq_threshold: float = 0.7,
        max_components: Optional[int] = None,
    ) -> "Sparc":
        """
        Analyze spectra for outlier detection and clustering.

        Args:
            contamination: Contamination parameter for outlier detection
            freq_threshold: Frequency threshold for noise detection
            max_components: Maximum number of components for clustering

        Returns:
            Self for method chaining
        """
        if self.roi_spectra is None:
            raise ValueError("No ROI spectra available. Call filter_rois() first.")

        # Extract non-Bayer spectra for analysis (same as original)
        nb_sort_indices = np.argsort(WLS[3:]) + 3
        nb_spectra = self.roi_spectra[:, nb_sort_indices]

        # Detect outliers using frequency analysis
        self.outlier_mask = detect_unique_spectra(
            nb_spectra, contamination, freq_threshold
        )

        # Determine which spectra to use for clustering
        if len(nb_spectra[self.outlier_mask]) > 3:
            filtered_spectra = nb_spectra[self.outlier_mask]
            spectra_to_cluster = filtered_spectra
        else:
            filtered_spectra = nb_spectra
            spectra_to_cluster = nb_spectra
            # Set outlier mask to all True if not enough outliers found (matches original)
            self.outlier_mask = np.ones(len(nb_spectra), dtype=bool)

        # Clustering analysis using Bayesian GMM
        if max_components is None:
            max_components = min(9, len(spectra_to_cluster))

        self.clustering_result, self.all_clustering_results = auto_bayesian_gmm(
            spectra_to_cluster, max_components
        )

        self.config["spectral_analysis"] = {
            "contamination": contamination,
            "freq_threshold": freq_threshold,
            "max_components": max_components,
        }

        return self

    def select_final_rois(self) -> "Sparc":
        """
        Apply final selection heuristics to choose representative ROIs.

        Returns:
            Self for method chaining
        """
        if self.clustering_result is None:
            raise ValueError("No clustering results. Call analyze_spectra() first.")
        if self.area_filtered_rois is None:
            raise ValueError("No filtered ROIs. Call filter_rois() first.")

        # Get ROIs that correspond to the albedo-filtered spectra
        albedo_filtered_rois = self.area_filtered_rois[self.albedo_valid_indices]

        # Apply outlier mask to get ROIs corresponding to clustering input
        if len(self.roi_spectra[self.outlier_mask]) > 3:
            # Use outlier-filtered ROIs and data
            selected_rois = albedo_filtered_rois[self.outlier_mask]
            selected_stds = self.roi_stds[self.outlier_mask]
        else:
            # Use all albedo-filtered ROIs and data
            selected_rois = albedo_filtered_rois
            selected_stds = self.roi_stds

        # Apply selection heuristics to get final representative ROIs
        self.roi_indices = apply_selection_heuristics(
            selected_rois, selected_stds, self.clustering_result["labels"]
        )

        self.final_rois = selected_rois[self.roi_indices]

        return self

    def plot_results(self, show: bool = True) -> Tuple:
        """
        Plot the final results.

        Args:
            show: Whether to display the plots

        Returns:
            Tuple of (spectra_fig, roi_fig) matplotlib figures
        """
        if self.final_rois is None:
            raise ValueError("No final ROIs. Run the complete pipeline first.")

        # Get final spectra and stds that correspond to final ROIs
        if len(self.roi_spectra[self.outlier_mask]) > 3:
            # Use outlier-filtered spectra, then select by roi_indices
            outlier_spectra = self.roi_spectra[self.outlier_mask]
            outlier_stds = self.roi_stds[self.outlier_mask]
            final_spectra = outlier_spectra[self.roi_indices]
            final_stds = outlier_stds[self.roi_indices]
        else:
            # Use all spectra, then select by roi_indices
            final_spectra = self.roi_spectra[self.roi_indices]
            final_stds = self.roi_stds[self.roi_indices]

        # Create plots
        spectra_fig = plot_spectra(final_spectra, final_stds, show=show)
        roi_fig = plot_roi_image(
            self.load_result["rgb_img"], self.final_rois, show=show
        )

        return spectra_fig, roi_fig

    def run_pipeline(
        self, iof_path: str, seq_id: Optional[str] = None, obs_ix: int = 0, **kwargs
    ) -> "Sparc":
        """
        Run the complete SPARC pipeline.

        Args:
            iof_path: Path to IOF data
            seq_id: Sequence ID (optional)
            obs_ix: Observation index
            **kwargs: Additional configuration parameters

        Returns:
            Self for method chaining
        """
        # Extract configuration
        load_kwargs = kwargs.get("load_kwargs", {})
        preprocess_kwargs = kwargs.get("preprocess_kwargs", {})
        roi_kwargs = kwargs.get("roi_kwargs", {})
        area_threshold = kwargs.get("area_threshold", DEFAULT_ROI_AREA_THRESHOLD)
        spectral_kwargs = kwargs.get("spectral_kwargs", {})

        # Run pipeline
        (
            self.load_data(iof_path, seq_id, obs_ix, **load_kwargs)
            .preprocess(**preprocess_kwargs)
            .segment()
            .extract_rois(**roi_kwargs)
            .filter_rois(area_threshold)
            .analyze_spectra(**spectral_kwargs)
            .select_final_rois()
        )

        return self

    @property
    def is_complete(self) -> bool:
        """Check if the pipeline has been completed."""
        return self.final_rois is not None

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the current pipeline state."""
        summary = {
            "data_loaded": self.load_result is not None,
            "preprocessed": self.processed_data is not None,
            "segmented": self.segments is not None,
            "rois_extracted": self.unfiltered_rois is not None,
            "rois_filtered": self.area_filtered_rois is not None,
            "spectra_analyzed": self.clustering_result is not None,
            "final_selection": self.final_rois is not None,
            "pipeline_complete": self.is_complete,
        }

        if self.load_result:
            summary["scene_id"] = self.load_result["id"]
        if self.unfiltered_rois is not None:
            summary["total_rois"] = len(self.unfiltered_rois)
        if self.area_filtered_rois is not None:
            summary["filtered_rois"] = len(self.area_filtered_rois)
        if self.final_rois is not None:
            summary["final_rois"] = len(self.final_rois)
        if self.clustering_result:
            summary["spectral_clusters"] = self.clustering_result["n_components"]

        return summary

    def export_results(
        self,
        output_dir: str,
        base_name: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export all results using SparcExporter.

        Args:
            output_dir: Output directory for results
            base_name: Base name for files (default: scene_id)
            metadata: Optional metadata for CSV export

        Returns:
            Dictionary mapping export type to file path
        """
        from ..utils.io import SparcExporter

        exporter = SparcExporter(self)
        return exporter.export_complete_results(output_dir, base_name, metadata)

    def save_roi_csv(
        self, output_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save ROI results to CSV file in marslab format.

        Args:
            output_path: Output CSV file path
            metadata: Optional metadata dictionary
        """
        from ..utils.io import SparcExporter

        exporter = SparcExporter(self)
        exporter.save_roi_csv(output_path, metadata)

    def save_plots(
        self, output_dir: str, base_name: str = None, dpi: int = 300
    ) -> Dict[str, str]:
        """
        Save all plots to files.

        Args:
            output_dir: Output directory
            base_name: Base name for files
            dpi: Image resolution

        Returns:
            Dictionary of saved plot paths
        """
        from ..utils.io import SparcExporter
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            base_name = self.load_result["id"] if self.load_result else "sparc_results"

        exporter = SparcExporter(self)
        saved_plots = {}

        # Save spectra plot
        spectra_path = output_dir / f"{base_name}_spectra.png"
        exporter.save_spectra_plot(spectra_path, dpi=dpi)
        saved_plots["spectra"] = str(spectra_path)

        # Save ROI context image
        context_path = output_dir / f"{base_name}_roi_context.png"
        exporter.save_roi_context_image(context_path, dpi=dpi)
        saved_plots["roi_context"] = str(context_path)

        # Save pipeline summary
        summary_path = output_dir / f"{base_name}_pipeline_summary.png"
        exporter.save_pipeline_summary_image(summary_path, dpi=dpi)
        saved_plots["pipeline_summary"] = str(summary_path)

        return saved_plots

    @classmethod
    def from_saved_results(
        cls, file_path: Union[str, Path], sam_model_path: str
    ) -> "Sparc":
        """
        Create a Sparc instance from previously saved results.

        Args:
            file_path: Path to saved results file (.pkl, .json, or .npz)
            sam_model_path: Path to SAM model weights

        Returns:
            Sparc instance with restored state
        """
        from ..utils.io import load_sparc_results

        # Load saved results
        saved_data = load_sparc_results(file_path)

        # Create new instance
        instance = cls(sam_model_path)

        # Restore state
        instance.load_saved_state(saved_data)

        print(f"Restored SPARC instance from: {file_path}")
        if instance.load_result:
            print(f"   Scene ID: {instance.load_result.get('id', 'unknown')}")
        print(f"   Pipeline complete: {instance.is_complete}")

        return instance

    def reload_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Reload state from file into current instance.

        Args:
            file_path: Path to saved results file
        """
        from ..utils.io import load_sparc_results

        saved_data = load_sparc_results(file_path)
        self.load_saved_state(saved_data)
        self._current_load_path = str(file_path)

    @property
    def is_loaded_from_file(self) -> bool:
        """Check if instance was loaded from a saved file."""
        return getattr(self, "_loaded_from_file", False)
