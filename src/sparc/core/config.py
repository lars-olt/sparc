"""Configuration dataclasses for SPARC pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class SegmentationBackend(Enum):
    """Segmentation backend options."""
    CPU = "cpu"
    GPU = "gpu"
    OPTIMIZED = "optimized"


class ROIBackend(Enum):
    """ROI extraction backend options."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"


@dataclass
class LoadConfig:
    """Configuration for data loading."""
    iof_path: str
    seq_id: Optional[str] = None
    obs_ix: int = 0
    do_apply_pixmaps: bool = True
    ignore_bayers: bool = False


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    shadow_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'percentiles': (20, 100),
        'operator': 'and'
    })
    skymask_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'percentile': 75,
        'edge_params': {"maximum": 5, "erosion": 3},
        'input_median': 5,
        'trace_maximum': 5,
        'cutoffs': {
            "extent": 0.05, "coverage": None, "v": 0.9, "h": None
        },
        'input_mask_dilation': None,
        'input_stretch': (10, 1),
        'floodfill': True,
        'trim_params': {"trim": False},
        'clear': True,
        'colorblock': False,
        'respect_mask': False,
    })
    apply_r_star: bool = True


@dataclass
class SegmentConfig:
    """Configuration for segmentation."""
    sam_model_path: str
    backend: SegmentationBackend = SegmentationBackend.CPU
    preserve_background: bool = False
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    model_type: Optional[str] = None


@dataclass
class ROIConfig:
    """Configuration for ROI extraction and filtering."""
    backend: ROIBackend = ROIBackend.SEQUENTIAL
    edge_offset: int = 10
    allowed_variance: float = 1.0
    area_threshold: int = 50
    albedo_ratio_threshold: float = 0.80
    min_segment_size: int = 50
    
    # Advanced Clustering Parameters
    min_cluster_area: int = 500         # Min area to attempt sub-clustering
    min_clean_area: int = 4000          # Min area after cleaning to sub-cluster
    morph_opening_threshold: int = 1000 # Area threshold to apply morph opening
    max_subclusters: int = 10           # Absolute max sub-clusters per segment
    subcluster_area_divisor: int = 1000 # Divisor for density-based clustering limits


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis."""
    contamination: float = 0.1
    freq_threshold: float = 0.7
    max_components: Optional[int] = None


@dataclass
class PerformanceConfig:
    """Configuration for performance options."""
    n_threads: Optional[int] = None
    use_gpu: bool = False


@dataclass
class SparcConfig:
    """Complete SPARC pipeline configuration."""
    load: LoadConfig
    segment: SegmentConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    def validate(self):
        """Validate configuration consistency."""
        if self.roi.backend == ROIBackend.THREADED and self.performance.n_threads is None:
            import psutil
            self.performance.n_threads = max(1, psutil.cpu_count(logical=False) - 1)
        
        if self.segment.backend == SegmentationBackend.GPU or self.segment.backend == SegmentationBackend.OPTIMIZED:
            try:
                import torch
                if not torch.cuda.is_available():
                    self.segment.backend = SegmentationBackend.CPU
            except ImportError:
                self.segment.backend = SegmentationBackend.CPU