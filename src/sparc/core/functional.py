"""Functional entry points for the SPARC pipeline."""

from typing import Any, Dict, Optional

from .config import SparcConfig, LoadConfig, SegmentConfig
from .state import SparcState
from .result import SparcResult
from .pipeline import (
    load_step, preprocess_step, segment_step,
    roi_step, spectral_step, selection_step,
)
from .constants import get_instrument_config
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def run_sparc(iof_path: str,
              sam_model_path: str,
              config: Optional[SparcConfig] = None) -> SparcResult:
    """Run the full SPARC pipeline from scratch."""
    if config is None:
        config = SparcConfig(
            load    = LoadConfig(iof_path=iof_path),
            segment = SegmentConfig(sam_model_path=sam_model_path),
        )
    config.validate()

    logger.info("Starting SPARC pipeline")
    state = SparcState()
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
    Run SPARC starting from an already-loaded scene, skipping load_step.
    Used by the GUI to avoid paying the IO cost twice.
    """
    config.validate()

    instrument        = load_result.get('instrument', 'ZCAM')
    instrument_config = get_instrument_config(instrument)
    instrument_config['wavelengths'] = load_result['bandset']._sparc_wavelengths

    state                   = SparcState()
    state.load_result       = load_result
    state.instrument_config = instrument_config
    state.using_pixmaps     = load_result.get('using_pixmaps', True)

    logger.info(f"Resuming SPARC from pre-loaded scene: {load_result['id']}")
    state = preprocess_step(state, config)
    state = segment_step(state, config)
    state = roi_step(state, config)
    state = spectral_step(state, config)
    state = selection_step(state, config)
    logger.info("Pipeline complete")

    return SparcResult.from_state(state)


def run_sparc_steps(config: SparcConfig) -> SparcState:
    """Run the full pipeline and return the raw state - useful for debugging."""
    config.validate()
    state = SparcState()
    state = load_step(state, config)
    state = preprocess_step(state, config)
    state = segment_step(state, config)
    state = roi_step(state, config)
    state = spectral_step(state, config)
    state = selection_step(state, config)
    return state