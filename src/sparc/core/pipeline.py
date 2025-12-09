"""Pure pipeline step functions for SPARC processing."""

import numpy as np
import time
from typing import Tuple

from .config import SparcConfig
from .state import SparcState
from .backends import dispatch_segmentation, dispatch_roi_extraction
from .logging_utils import setup_logger

from ..data.loading import load_cube
from ..preprocessing.masking import apply_masking
from ..preprocessing.calibration import apply_photometric_calibration
from ..roi.filtering import filter_by_area, filter_by_albedo_ratio, select_representative_rois
from ..spectral.analysis import detect_outlier_spectra, cluster_with_bayesian_gmm
from ..spectral.metrics import compute_roi_spectra
from ..utils.geometry import convert_to_plot_coords
from ..core.constants import WAVELENGTHS

logger = setup_logger(__name__)


def load_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Load hyperspectral data cube."""
    logger.info(f"Loading data from {config.load.iof_path}")
    t0 = time.time()
    
    load_result = load_cube(
        iof_path=config.load.iof_path,
        seq_id=config.load.seq_id,
        obs_ix=config.load.obs_ix,
        do_apply_pixmaps=config.load.do_apply_pixmaps,
        ignore_bayers=config.load.ignore_bayers
    )
    
    state.load_result = load_result
    state.using_pixmaps = config.load.do_apply_pixmaps
    
    logger.info(f"Loaded scene: {load_result['id']}")
    logger.debug(f"Data loading took {time.time() - t0:.2f}s")
    logger.debug(f"Merged cube shape: {load_result['cube'].shape}")
    logger.debug(f"Left cube shape: {load_result['left_cube'].shape}")
    logger.debug(f"Right cube shape: {load_result['right_cube'].shape}")
    
    return state


def preprocess_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Apply masking and photometric calibration."""
    if state.load_result is None:
        raise ValueError("No data loaded. Run load_step first.")
    
    logger.info("Preprocessing data (masking and calibration)")
    t0 = time.time()
    
    # Apply masking
    mask_result = apply_masking(
        state.load_result,
        state.using_pixmaps,
        config.preprocess.shadow_kwargs,
        config.preprocess.skymask_kwargs
    )
    
    # Store merged cube masks
    state.processed_data = mask_result['masked_cube']
    state.shadow_mask = mask_result['shadow_mask']
    state.sky_mask = mask_result['sky_mask']
    state.full_mask = mask_result['full_mask']
    
    # Store left/right masks
    state.left_shadow_mask = mask_result['left_shadow_mask']
    state.left_sky_mask = mask_result['left_sky_mask']
    state.left_full_mask = mask_result['left_full_mask']
    state.right_shadow_mask = mask_result['right_shadow_mask']
    state.right_sky_mask = mask_result['right_sky_mask']
    state.right_full_mask = mask_result['right_full_mask']
    
    # Logging mask statistics
    total_pixels = state.full_mask.size
    shadow_pct = np.count_nonzero(state.shadow_mask) / total_pixels * 100
    sky_pct = np.count_nonzero(state.sky_mask) / total_pixels * 100
    valid_pct = 100 - (np.count_nonzero(state.full_mask) / total_pixels * 100)
    
    logger.debug(f"Mask Stats: Shadow={shadow_pct:.1f}%, Sky={sky_pct:.1f}%")
    logger.debug(f"Valid pixels remaining: {valid_pct:.1f}%")

    # Apply photometric calibration
    state.photometrically_calibrated = apply_photometric_calibration(
        state.processed_data,
        state.load_result['bandset'].metadata,
        config.preprocess.apply_r_star
    )
    
    if logger.isEnabledFor(10):  # DEBUG level check
        # Calculate stats on valid pixels only
        valid_data = state.photometrically_calibrated.compressed()
        if valid_data.size > 0:
            logger.debug(f"Calibrated Data Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
            logger.debug(f"Mean Reflectance: {valid_data.mean():.3f}")
    
    logger.debug(f"Preprocessing took {time.time() - t0:.2f}s")
    return state


def segment_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Segment RGB image using SAM."""
    if state.load_result is None:
        raise ValueError("No data loaded. Run load_step first.")
    
    logger.info(f"Segmenting image using {config.segment.backend.value} backend")
    t0 = time.time()
    
    rgb_img = state.load_result['rgb_img']
    
    state.segments = dispatch_segmentation(
        model_path=config.segment.sam_model_path,
        img=rgb_img,
        backend=config.segment.backend,
        preserve_background=config.segment.preserve_background,
        points_per_side=config.segment.points_per_side,
        pred_iou_thresh=config.segment.pred_iou_thresh,
        model_type=config.segment.model_type
    )
    
    n_segments = len(np.unique(state.segments))
    logger.info(f"Found {n_segments} segments")
    logger.debug(f"Segmentation took {time.time() - t0:.2f}s")
    
    return state


def roi_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Extract and filter regions of interest."""
    if state.segments is None:
        raise ValueError("No segmentation. Run segment_step first.")
    if state.photometrically_calibrated is None:
        raise ValueError("No preprocessed data. Run preprocess_step first.")
    
    logger.info(f"Extracting ROIs using {config.roi.backend.value} backend")
    t0 = time.time()
    
    # Extract potential ROIs
    state.unfiltered_rois = dispatch_roi_extraction(
        segmented_img=state.segments,
        masked_cube=state.photometrically_calibrated,
        edge_offset=config.roi.edge_offset,
        allowed_variance=config.roi.allowed_variance,
        backend=config.roi.backend,
        n_threads=config.performance.n_threads,
        min_segment_size=config.roi.min_segment_size,
        min_cluster_area=config.roi.min_cluster_area,
        min_clean_area=config.roi.min_clean_area,
        morph_opening_threshold=config.roi.morph_opening_threshold,
        max_subclusters=config.roi.max_subclusters,
        subcluster_area_divisor=config.roi.subcluster_area_divisor
    )
    
    logger.info(f"Found {len(state.unfiltered_rois)} potential ROIs")
    
    # Filter by area
    state.area_filtered_rois = filter_by_area(
        state.unfiltered_rois,
        config.roi.area_threshold
    )
    
    logger.debug(f"Dropped {len(state.unfiltered_rois) - len(state.area_filtered_rois)} ROIs < {config.roi.area_threshold}px")
    logger.info(f"Retained {len(state.area_filtered_rois)} ROIs after area filtering")
    
    # Calculate average spectra for ROIs
    plot_coords = convert_to_plot_coords(state.area_filtered_rois)
    roi_spectra, roi_stds = compute_roi_spectra(
        state.photometrically_calibrated,
        plot_coords
    )
    
    # Filter by albedo ratio
    filtered_spectra, filtered_stds, valid_indices = filter_by_albedo_ratio(
        roi_spectra,
        roi_stds,
        config.roi.albedo_ratio_threshold
    )
    
    state.roi_spectra = filtered_spectra
    state.roi_stds = filtered_stds
    state.albedo_valid_indices = valid_indices
    
    dropped_albedo = len(roi_spectra) - len(filtered_spectra)
    logger.debug(f"Dropped {dropped_albedo} ROIs due to L/R albedo mismatch (< {config.roi.albedo_ratio_threshold})")
    logger.info(f"Retained {len(state.roi_spectra)} ROIs after albedo filtering")
    logger.debug(f"ROI extraction took {time.time() - t0:.2f}s")
    
    return state


def spectral_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Analyze spectra for outliers and clustering."""
    if state.roi_spectra is None:
        raise ValueError("No ROI spectra. Run roi_step first.")
    
    logger.info("Analyzing spectra for outliers and clustering")
    
    # Extract non-Bayer spectra
    n_bands = state.roi_spectra.shape[1]
    n_non_bayer = n_bands - 3
    available_wavelengths = WAVELENGTHS[3:3 + n_non_bayer]
    nb_sort_indices = np.argsort(available_wavelengths)
    nb_spectra = state.roi_spectra[:, 3:][:, nb_sort_indices]
    
    # Detect outliers
    state.outlier_mask = detect_outlier_spectra(
        nb_spectra,
        config.spectral.contamination,
        config.spectral.freq_threshold
    )
    
    n_outliers = np.count_nonzero(state.outlier_mask)
    logger.info(f"Detected {n_outliers} unique/outlier spectra")
    
    # Determine spectra for clustering
    if n_outliers > 3:
        spectra_to_cluster = nb_spectra[state.outlier_mask]
        logger.debug(f"Clustering subset of {n_outliers} outlier spectra")
    else:
        spectra_to_cluster = nb_spectra
        state.outlier_mask = np.ones(len(nb_spectra), dtype=bool)
        logger.debug("Too few outliers; clustering entire dataset")
    
    # Clustering analysis
    max_components = config.spectral.max_components
    if max_components is None:
        max_components = min(9, len(spectra_to_cluster))
    
    state.clustering_result, state.all_clustering_results = cluster_with_bayesian_gmm(
        spectra_to_cluster,
        max_components
    )
    
    n_clusters = state.clustering_result['n_components']
    method = state.clustering_result.get('model', 'unknown')
    logger.info(f"Found {n_clusters} spectral clusters")
    logger.debug(f"Optimal clustering method: {method}")
    logger.debug(f"Log-likelihood: {state.clustering_result['log_likelihood']:.2f}")
    
    return state


def selection_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Select final representative ROIs using heuristics."""
    if state.clustering_result is None:
        raise ValueError("No clustering results. Run spectral_step first.")
    
    logger.info("Selecting final representative ROIs")
    
    albedo_filtered_rois = state.area_filtered_rois[state.albedo_valid_indices]
    
    # Apply outlier mask
    if len(state.roi_spectra[state.outlier_mask]) > 3:
        selected_rois = albedo_filtered_rois[state.outlier_mask]
        selected_stds = state.roi_stds[state.outlier_mask]
        selected_spectra = state.roi_spectra[state.outlier_mask]
    else:
        selected_rois = albedo_filtered_rois
        selected_stds = state.roi_stds
        selected_spectra = state.roi_spectra
    
    # Apply selection heuristics
    state.roi_indices = select_representative_rois(
        selected_rois,
        selected_stds,
        state.clustering_result['labels']
    )
    
    state.final_rois = selected_rois[state.roi_indices]
    state.final_spectra = selected_spectra[state.roi_indices]
    state.final_stds = selected_stds[state.roi_indices]
    
    logger.info(f"Selected {len(state.final_rois)} final ROIs")
    logger.debug(f"Final selection reduced candidate pool from {len(selected_rois)} to {len(state.final_rois)}")
    
    return state