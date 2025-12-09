"""Functional entry point for SPARC pipeline."""

import logging
from typing import Optional

from .config import SparcConfig, LoadConfig, SegmentConfig
from .state import SparcState
from .result import SparcResult
from .pipeline import (
    load_step,
    preprocess_step,
    segment_step,
    roi_step,
    spectral_step,
    selection_step
)
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def run_sparc(iof_path: str,
              sam_model_path: str,
              config: Optional[SparcConfig] = None) -> SparcResult:
    """
    Run complete SPARC pipeline functionally.
    
    This is the main functional entry point for SPARC processing.
    It executes the entire pipeline from loading to final ROI selection.
    
    Args:
        iof_path: Path to IOF data directory
        sam_model_path: Path to SAM model weights
        config: Optional SparcConfig (uses defaults if None)
        
    Returns:
        SparcResult containing final ROIs, spectra, and metadata
        
    Example:
        >>> result = run_sparc(
        ...     iof_path="/path/to/data",
        ...     sam_model_path="/path/to/sam_model.pth"
        ... )
        >>> print(f"Found {len(result.final_rois)} ROIs")
    """
    # Create default config if not provided
    if config is None:
        config = SparcConfig(
            load=LoadConfig(iof_path=iof_path),
            segment=SegmentConfig(sam_model_path=sam_model_path)
        )
    
    # Validate configuration
    config.validate()
    
    # Initialize state
    state = SparcState()
    
    # Execute pipeline steps
    logger.info("Starting SPARC pipeline")
    state = load_step(state, config)
    state = preprocess_step(state, config)
    state = segment_step(state, config)
    state = roi_step(state, config)
    state = spectral_step(state, config)
    state = selection_step(state, config)
    logger.info("Pipeline complete")
    
    # Convert to result
    result = SparcResult.from_state(state)
    
    return result


def run_sparc_steps(config: SparcConfig) -> SparcState:
    """
    Run SPARC pipeline and return full state (for debugging/inspection).
    
    Args:
        config: Complete pipeline configuration
        
    Returns:
        Final SparcState with all intermediate results
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
