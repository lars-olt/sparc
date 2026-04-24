"""Functional entry point for SPARC pipeline."""

from typing import Any, Dict, Optional

from .config import SparcConfig, LoadConfig, SegmentConfig
from .state import SparcState
from .result import SparcResult
from .pipeline import (
    load_step,
    preprocess_step,
    segment_step,
    roi_step,
    spectral_step,
    selection_step,
)
from .constants import get_instrument_config
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def run_sparc(iof_path: str,
              sam_model_path: str,
              config: Optional[SparcConfig] = None) -> SparcResult:
    """
    Run complete SPARC pipeline from scratch.

    Args:
        iof_path:       Path to IOF data directory.
        sam_model_path: Path to SAM model weights.
        config:         Optional SparcConfig (uses defaults if None).

    Returns:
        SparcResult containing final ROIs, spectra, and metadata.
    """
    if config is None:
        config = SparcConfig(
            load=LoadConfig(iof_path=iof_path),
            segment=SegmentConfig(sam_model_path=sam_model_path),
        )

    config.validate()
    state = SparcState()

    logger.info("Starting SPARC pipeline")
    state = load_step(state, config)
    state = preprocess_step(state, config)
    state = segment_step(state, config)
    state = roi_step(state, config)
    state = spectral_step(state, config)
    state = selection_step(state, config)
    logger.info("Pipeline complete")

    return SparcResult.from_state(state)


def run_sparc_from_load_result(load_result: Dict[str, Any],
                               config: SparcConfig) -> SparcResult:
    """
    Run SPARC pipeline using a pre-loaded load_result, skipping load_step.

    Use this when the scene has already been loaded by the GUI so we don't
    pay the IO cost twice.

    Args:
        load_result: Dict returned by sparc.data.loading.load_cube.
        config:      Complete SparcConfig. config.load fields are ignored
                     since loading is already done.

    Returns:
        SparcResult containing final ROIs, spectra, and metadata.
    """
    config.validate()

    # Reconstruct exactly what load_step would have written onto state.
    instrument = load_result.get('instrument', 'ZCAM')
    instrument_config = get_instrument_config(instrument)
    instrument_config['wavelengths'] = load_result['bandset']._sparc_wavelengths

    state = SparcState()
    state.load_result       = load_result
    state.instrument_config = instrument_config
    state.using_pixmaps     = load_result.get('using_pixmaps', True)

    logger.info(f"Resuming SPARC pipeline from pre-loaded scene: {load_result['id']}")
    state = preprocess_step(state, config)
    state = segment_step(state, config)
    state = roi_step(state, config)
    state = spectral_step(state, config)
    state = selection_step(state, config)
    logger.info("Pipeline complete")

    return SparcResult.from_state(state)


def run_sparc_steps(config: SparcConfig) -> SparcState:
    """
    Run SPARC pipeline and return full state (for debugging/inspection).

    Args:
        config: Complete pipeline configuration.

    Returns:
        Final SparcState with all intermediate results.
    """
    config.validate()
    state = SparcState()

    state = load_step(state, config)
    state = preprocess_step(state, config)
    state = segment_step(state, config)
    state = roi_step(state, config)
    state = spectral_step(state, config)
    state = selection_step(state, config)

    return state