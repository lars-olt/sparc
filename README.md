# SPARC: Spectral Pattern Analysis for ROI Classification

A pipeline for automatically extracting regions of interest from Mars rover hyperspectral images. SPARC combines deep learning segmentation with spectral clustering to identify geologically releant features in MastcamZ data.

## Architecture

SPARC separates state management from pure data transformations:

* Shell ([sparc.py](src/sparc/core/sparc.py)): Object-oriented interface for state management and configuration
* Core ([pipeline.py](src/sparc/core/pipeline.py)): Pure functions that transform data through each processing step
* State ([state.py](src/sparc/core/state.py)): Dataclass tracking the pipeline lifecycle
* Backends ([backends.py](src/sparc/core/backends.py)): Abstraction layer for quickly swapping segmentation and clustering implementations

## Installation

### 1. Clone and install

```bash
git clone https://github.com/lars-olt/sparc.git
cd sparc
uv venv
uv sync
```

### 2. GPU acceleration (optional)

SPARC runs in CPU mode by default. For faster segmentation, install PyTorch with CUDA support on top of the uv environment. Find the right command for your system and CUDA version at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then run it with `--force-reinstall`:

```bash
# example for CUDA 12.1 - replace cu121 with your version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 3. SAM model checkpoint

Download `sam_vit_h_4b8939.pth` from the [Segment Anything repository](https://github.com/facebookresearch/segment-anything) and note its path — you will pass it to SPARC at runtime via `sam_model_path`.

## Quick Start

The primary entry point is the `Sparc` class.

```python
from sparc import Sparc

# Initialize
sparc = Sparc(
    sam_model_path="./models/sam_vit_h_4b8939.pth",
    use_gpu=True,
    use_threading=True,
    verbose=True
)

# Run pipeline
(sparc
    .load(iof_path="/path/to/mcz/data", obs_ix=0)
    .preprocess(apply_r_star=True)
    .segment(points_per_side=32)
    .extract_rois(
        area_threshold=50,
        min_cluster_area=500,
        min_clean_area=4000
    )
    .analyze(max_components=9)
    .select()
)

# Visualize results
sparc.plot(figsize=(15, 12))

# Access results
result = sparc.result
print(f"Found {len(result.final_rois)} ROIs in {result.n_clusters} spectral clusters")

# Export
from sparc.core.result import export_sel
export_sel(result, "output/scene.sel")
```

SPARC also supports Pancam data - pass `instrument="PCAM"` to `.load()`.

### Functional API

For scripting or batch use, a functional entry point is also available:

```python
from sparc.core.functional import run_sparc
from sparc.core.config import SparcConfig, LoadConfig, SegmentConfig

config = SparcConfig(
    load    = LoadConfig(iof_path="/path/to/data", instrument="ZCAM"),
    segment = SegmentConfig(sam_model_path="./models/sam_vit_h_4b8939.pth"),
)
result = run_sparc(config=config)
```

### Configuration

Configuration is handled via strongly-typed dataclasses in [`config.py`](src/sparc/core/config.py). Parameters can be set via method arguments as shown above, or by modifying the config object directly before running individual steps.

Key tuning parameters:

- `albedo_ratio_threshold` (default: `0.80`) - filters ROIs with a large brightness discrepancy between left and right cameras. ZCAM only.
- `allowed_variance` (default: `1.0`) - threshold for splitting a SAM segment into multiple spectral subclusters. Lower values produce finer splits.
- `edge_offset` (default: `10`) - pixels ignored around the image border to avoid edge artifacts.
- `max_subclusters` (default: `10`) - hard limit on subclusters per segment to prevent fragmentation.
- `max_components` (default: `9`) - maximum number of spectral clusters the Bayesian GMM may find.
