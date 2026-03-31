"""SPARC pipeline steps - load, preprocess, segment, extract, analyze, select."""

import time

import numpy as np

from .config import SparcConfig
from .state import SparcState
from .backends import dispatch_segmentation, dispatch_roi_extraction
from .logging_utils import setup_logger
from .constants import WAVELENGTHS, get_instrument_config

from ..data.loading import load_cube
from ..preprocessing.masking import apply_masking
from ..preprocessing.calibration import apply_photometric_calibration
from ..roi.filtering import filter_by_area, filter_by_albedo_ratio, select_representative_rois
from ..spectral.analysis import detect_outlier_spectra, cluster_with_bayesian_gmm
from ..spectral.metrics import compute_roi_spectra
from ..utils.geometry import convert_to_plot_coords

logger = setup_logger(__name__)


def load_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Load the hyperspectral cube and attach instrument configuration."""
    logger.info(f"Loading {config.load.instrument} data from {config.load.iof_path}")
    t0 = time.time()

    state.instrument_config = get_instrument_config(config.load.instrument)
    load_result = load_cube(
        iof_path        = config.load.iof_path,
        instrument      = config.load.instrument,
        seq_id          = config.load.seq_id,
        obs_ix          = config.load.obs_ix,
        do_apply_pixmaps= config.load.do_apply_pixmaps,
        ignore_bayers   = config.load.ignore_bayers,
        rgb_bands       = config.load.rgb_bands,
    )

    # Wavelengths are set during loading to match the exact cube band order.
    state.instrument_config['wavelengths'] = load_result['bandset']._sparc_wavelengths
    state.load_result   = load_result
    state.using_pixmaps = config.load.do_apply_pixmaps

    logger.info(f"Loaded scene: {load_result['id']}")
    logger.debug(f"Load took {time.time() - t0:.2f}s | cube shape: {load_result['cube'].shape}")
    logger.debug(f"Wavelengths: {state.instrument_config['wavelengths']}")
    return state


def preprocess_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Apply shadow/sky masking and photometric calibration."""
    if state.load_result is None:
        raise ValueError("No data loaded. Run load_step first.")

    logger.info("Preprocessing - masking and photometric calibration")
    t0 = time.time()

    mask_result = apply_masking(
        state.load_result,
        state.using_pixmaps,
        config.preprocess.shadow_kwargs,
        config.preprocess.skymask_kwargs,
    )

    state.processed_data     = mask_result['masked_cube']
    state.shadow_mask        = mask_result['shadow_mask']
    state.sky_mask           = mask_result['sky_mask']
    state.full_mask          = mask_result['full_mask']
    state.left_shadow_mask   = mask_result['left_shadow_mask']
    state.left_sky_mask      = mask_result['left_sky_mask']
    state.left_full_mask     = mask_result['left_full_mask']
    state.right_shadow_mask  = mask_result['right_shadow_mask']
    state.right_sky_mask     = mask_result['right_sky_mask']
    state.right_full_mask    = mask_result['right_full_mask']

    total = state.full_mask.size
    logger.debug(
        f"Shadow={np.count_nonzero(state.shadow_mask)/total*100:.1f}%  "
        f"Sky={np.count_nonzero(state.sky_mask)/total*100:.1f}%  "
        f"Valid={100 - np.count_nonzero(state.full_mask)/total*100:.1f}%"
    )

    # Pancam stores its PDS label on the bandset; ZCAM uses bandset.metadata.
    bandset = state.load_result['bandset']
    meta    = getattr(bandset, '_sparc_label', None) or bandset.metadata

    state.photometrically_calibrated = apply_photometric_calibration(
        state.processed_data,
        meta,
        config.preprocess.apply_r_star,
    )

    if logger.isEnabledFor(10):
        valid = state.photometrically_calibrated.compressed()
        if valid.size > 0:
            logger.debug(f"R* range: [{valid.min():.3f}, {valid.max():.3f}]  mean: {valid.mean():.3f}")

    logger.debug(f"Preprocessing took {time.time() - t0:.2f}s")
    return state


def segment_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Segment the RGB image with SAM."""
    if state.load_result is None:
        raise ValueError("No data loaded. Run load_step first.")

    logger.info(f"Segmenting with {config.segment.backend.value}")
    t0 = time.time()

    state.segments = dispatch_segmentation(
        model_path          = config.segment.sam_model_path,
        img                 = state.load_result['rgb_img'],
        backend             = config.segment.backend,
        preserve_background = config.segment.preserve_background,
        points_per_side     = config.segment.points_per_side,
        pred_iou_thresh     = config.segment.pred_iou_thresh,
        model_type          = config.segment.model_type,
    )

    logger.info(f"Found {len(np.unique(state.segments))} segments in {time.time() - t0:.2f}s")
    return state


def roi_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Extract ROIs from segments and filter by area and albedo."""
    if state.segments is None:
        raise ValueError("No segmentation. Run segment_step first.")
    if state.photometrically_calibrated is None:
        raise ValueError("No calibrated data. Run preprocess_step first.")

    logger.info(f"Extracting ROIs with {config.roi.backend.value}")
    t0 = time.time()

    state.unfiltered_rois = dispatch_roi_extraction(
        segmented_img          = state.segments,
        masked_cube            = state.photometrically_calibrated,
        edge_offset            = config.roi.edge_offset,
        allowed_variance       = config.roi.allowed_variance,
        backend                = config.roi.backend,
        n_threads              = config.performance.n_threads,
        min_segment_size       = config.roi.min_segment_size,
        min_cluster_area       = config.roi.min_cluster_area,
        min_clean_area         = config.roi.min_clean_area,
        morph_opening_threshold= config.roi.morph_opening_threshold,
        max_subclusters        = config.roi.max_subclusters,
        subcluster_area_divisor= config.roi.subcluster_area_divisor,
    )

    state.area_filtered_rois = filter_by_area(state.unfiltered_rois, config.roi.area_threshold)
    logger.info(
        f"{len(state.unfiltered_rois)} ROIs found: "
        f"{len(state.area_filtered_rois)} after area filter"
    )

    roi_spectra, roi_stds = compute_roi_spectra(
        state.photometrically_calibrated,
        convert_to_plot_coords(state.area_filtered_rois),
    )

    # Albedo ratio filtering is ZCAM-specific - Pancam L/R geometry differs.
    if state.load_result.get('instrument', 'ZCAM') == 'ZCAM':
        filtered_spectra, filtered_stds, valid_indices = filter_by_albedo_ratio(
            roi_spectra, roi_stds, config.roi.albedo_ratio_threshold
        )
    else:
        filtered_spectra = roi_spectra
        filtered_stds    = roi_stds
        valid_indices    = np.arange(len(roi_spectra))

    state.roi_spectra          = filtered_spectra
    state.roi_stds             = filtered_stds
    state.albedo_valid_indices = valid_indices

    logger.info(f"{len(state.roi_spectra)} ROIs retained after albedo filter")
    logger.debug(f"ROI extraction took {time.time() - t0:.2f}s")
    return state


def spectral_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Detect spectrally unique ROIs and cluster them."""
    if state.roi_spectra is None:
        raise ValueError("No ROI spectra. Run roi_step first.")

    logger.info("Spectral analysis - outlier detection and clustering")

    instrument = (state.instrument_config or {}).get('instrument', 'ZCAM')

    if instrument == 'ZCAM':
        # Hardcoded to preserve reproducibility with existing ZCAM results.
        n_non_bayer      = state.roi_spectra.shape[1] - 3
        avail_wls        = WAVELENGTHS[3:3 + n_non_bayer]
        nb_spectra       = state.roi_spectra[:, 3:][:, np.argsort(avail_wls)]
    else:
        bayer_cutoff     = state.instrument_config['bayer_cutoff']
        avail_wls        = state.instrument_config['wavelengths'][bayer_cutoff:]
        nb_spectra       = state.roi_spectra[:, bayer_cutoff:][:, np.argsort(avail_wls)]

    state.outlier_mask = detect_outlier_spectra(
        nb_spectra,
        config.spectral.contamination,
        config.spectral.freq_threshold,
    )

    n_outliers = np.count_nonzero(state.outlier_mask)
    logger.info(f"Detected {n_outliers} spectrally unique ROIs")

    if n_outliers > 3:
        spectra_to_cluster = nb_spectra[state.outlier_mask]
    else:
        spectra_to_cluster = nb_spectra
        state.outlier_mask = np.ones(len(nb_spectra), dtype=bool)
        logger.debug("Too few outliers - clustering full dataset")

    max_components = min(config.spectral.max_components, len(spectra_to_cluster))

    state.clustering_result, state.all_clustering_results = cluster_with_bayesian_gmm(
        spectra_to_cluster, max_components
    )

    logger.info(f"Found {state.clustering_result['n_components']} spectral clusters")
    logger.debug(f"Method: {state.clustering_result.get('model')}  "
                 f"LL: {state.clustering_result['log_likelihood']:.2f}")
    return state


def selection_step(state: SparcState, config: SparcConfig) -> SparcState:
    """Select one representative ROI per spectral cluster."""
    if state.clustering_result is None:
        raise ValueError("No clustering results. Run spectral_step first.")

    logger.info("Selecting representative ROIs")

    albedo_filtered_rois = state.area_filtered_rois[state.albedo_valid_indices]

    if len(state.roi_spectra[state.outlier_mask]) > 3:
        selected_rois    = albedo_filtered_rois[state.outlier_mask]
        selected_stds    = state.roi_stds[state.outlier_mask]
        selected_spectra = state.roi_spectra[state.outlier_mask]
    else:
        selected_rois    = albedo_filtered_rois
        selected_stds    = state.roi_stds
        selected_spectra = state.roi_spectra

    state.roi_indices    = select_representative_rois(
        selected_rois, selected_stds, state.clustering_result['labels']
    )
    state.final_rois     = selected_rois[state.roi_indices]
    state.final_spectra  = selected_spectra[state.roi_indices]
    state.final_stds     = selected_stds[state.roi_indices]

    logger.info(
        f"Selected {len(state.final_rois)} final ROIs "
        f"from {len(selected_rois)} candidates"
    )
    return state