"""Simple OO shell for SPARC pipeline - delegates all work to pure functions."""

from typing import Optional
import matplotlib.pyplot as plt

from .config import (
    SparcConfig,
    LoadConfig,
    PreprocessConfig,
    SegmentConfig,
    ROIConfig,
    SpectralConfig,
    PerformanceConfig,
    SegmentationBackend,
    ROIBackend
)
from .state import SparcState
from .result import SparcResult, plot_result
from .pipeline import (
    load_step,
    preprocess_step,
    segment_step,
    roi_step,
    spectral_step,
    selection_step
)
from .logging_utils import setup_logger, configure_logging

logger = setup_logger(__name__)


class Sparc:
    """
    Simple OO interface for SPARC pipeline.
    
    This class stores configuration and state, and delegates all processing
    to pure functions. It provides a clean, readable interface for interactive use.
    """
    
    def __init__(self,
                 sam_model_path: str,
                 use_gpu: bool = False,
                 use_threading: bool = False,
                 n_threads: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize SPARC with model path and performance options.
        
        Args:
            sam_model_path: Path to SAM model weights
            use_gpu: Enable GPU acceleration for segmentation
            use_threading: Enable threaded ROI extraction
            n_threads: Number of threads (None = auto-detect)
            verbose: Enable verbose logging (DEBUG level)
        """
        # Configure logging verbosity
        configure_logging(verbose)
        
        self.config = SparcConfig(
            load=LoadConfig(iof_path=""),  # Will be set in load()
            segment=SegmentConfig(
                sam_model_path=sam_model_path,
                backend=SegmentationBackend.GPU if use_gpu else SegmentationBackend.CPU
            ),
            preprocess=PreprocessConfig(),
            roi=ROIConfig(
                backend=ROIBackend.THREADED if use_threading else ROIBackend.SEQUENTIAL
            ),
            spectral=SpectralConfig(),
            performance=PerformanceConfig(
                n_threads=n_threads,
                use_gpu=use_gpu
            )
        )
        
        self.state = SparcState()
        self._result: Optional[SparcResult] = None
        
        if verbose:
            logger.debug("SPARC initialized with verbose logging enabled")
    
    def load(self,
             iof_path: str,
             seq_id: Optional[str] = None,
             obs_ix: int = 0,
             do_apply_pixmaps: bool = True,
             ignore_bayers: bool = False) -> 'Sparc':
        """Load hyperspectral data."""
        self.config.load = LoadConfig(
            iof_path=iof_path,
            seq_id=seq_id,
            obs_ix=obs_ix,
            do_apply_pixmaps=do_apply_pixmaps,
            ignore_bayers=ignore_bayers
        )
        
        self.state = load_step(self.state, self.config)
        return self
    
    def preprocess(self,
                   shadow_kwargs: Optional[dict] = None,
                   skymask_kwargs: Optional[dict] = None,
                   apply_r_star: bool = True) -> 'Sparc':
        """Preprocess data with masking and calibration."""
        if shadow_kwargs is not None:
            self.config.preprocess.shadow_kwargs = shadow_kwargs
        if skymask_kwargs is not None:
            self.config.preprocess.skymask_kwargs = skymask_kwargs
        self.config.preprocess.apply_r_star = apply_r_star
        
        self.state = preprocess_step(self.state, self.config)
        return self
    
    def segment(self,
                preserve_background: bool = False,
                points_per_side: int = 32,
                pred_iou_thresh: float = 0.88) -> 'Sparc':
        """Segment RGB image using SAM."""
        self.config.segment.preserve_background = preserve_background
        self.config.segment.points_per_side = points_per_side
        self.config.segment.pred_iou_thresh = pred_iou_thresh
        
        self.state = segment_step(self.state, self.config)
        return self
    
    def extract_rois(self,
                    edge_offset: int = 10,
                    allowed_variance: float = 1.0,
                    area_threshold: int = 50,
                    albedo_ratio_threshold: float = 0.80,
                    min_cluster_area: int = 500,
                    min_clean_area: int = 4000,
                    morph_opening_threshold: int = 1000,
                    max_subclusters: int = 10,
                    subcluster_area_divisor: int = 1000) -> 'Sparc':
        """Extract and filter regions of interest."""
        self.config.roi.edge_offset = edge_offset
        self.config.roi.allowed_variance = allowed_variance
        self.config.roi.area_threshold = area_threshold
        self.config.roi.albedo_ratio_threshold = albedo_ratio_threshold
        self.config.roi.min_cluster_area = min_cluster_area
        self.config.roi.min_clean_area = min_clean_area
        self.config.roi.morph_opening_threshold = morph_opening_threshold
        self.config.roi.max_subclusters = max_subclusters
        self.config.roi.subcluster_area_divisor = subcluster_area_divisor
        
        self.state = roi_step(self.state, self.config)
        return self
    
    def analyze(self,
                contamination: float = 0.1,
                freq_threshold: float = 0.7,
                max_components: Optional[int] = None) -> 'Sparc':
        """Analyze spectra for outliers and clustering."""
        self.config.spectral.contamination = contamination
        self.config.spectral.freq_threshold = freq_threshold
        self.config.spectral.max_components = max_components
        
        self.state = spectral_step(self.state, self.config)
        return self
    
    def select(self) -> 'Sparc':
        """Select final representative ROIs."""
        self.state = selection_step(self.state, self.config)
        self._result = SparcResult.from_state(self.state)
        return self
    
    def run_all(self,
                iof_path: str,
                seq_id: Optional[str] = None,
                obs_ix: int = 0) -> 'Sparc':
        """Run complete pipeline with defaults."""
        return (self
                .load(iof_path, seq_id, obs_ix)
                .preprocess()
                .segment()
                .extract_rois()
                .analyze()
                .select())
    
    def plot(self,
             show_segments: bool = True,
             show_rois: bool = True,
             show_spectra: bool = True,
             figsize: tuple = (15, 10)) -> plt.Figure:
        """Plot pipeline results."""
        if self._result is None:
            if self.state.final_rois is None:
                raise ValueError("Pipeline not complete. Run all steps first.")
            self._result = SparcResult.from_state(self.state)
        
        fig = plot_result(
            self._result,
            show_segments=show_segments,
            show_rois=show_rois,
            show_spectra=show_spectra,
            figsize=figsize
        )
        plt.show()
        return fig
    
    def get_result(self) -> SparcResult:
        """Get immutable result object."""
        if self._result is None:
            if self.state.final_rois is None:
                raise ValueError("Pipeline not complete. Run all steps first.")
            self._result = SparcResult.from_state(self.state)
        
        return self._result
    
    @property
    def result(self) -> SparcResult:
        """Convenience property to get result."""
        return self.get_result()