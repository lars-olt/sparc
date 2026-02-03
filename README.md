# SPARC: Spectral Pattern Analysis for ROI Classification

A pipeline for automatically extracting regions of interest from Mars rover hyperspectral images. SPARC combines deep learning segmentation with spectral clustering to identify geologically releant features in MastcamZ data.

## Architecture

SPARC separates state management from pure data transformations:

* Shell ([sparc.py](src/sparc/core/sparc.py)): Object-oriented interface for state management and configuration
* Core ([pipeline.py](src/sparc/core/pipeline.py)): Pure functions that transform data through each processing step
* State ([state.py](src/sparc/core/state.py)): Dataclass tracking the pipeline lifecycle
* Backends ([backends.py](src/sparc/core/backends.py)): Abstraction layer for quickly swapping segmentation and clustering implementations

## Installation & Setup

This project requires manual setup of several dependencies.

### 1. PyTorch & CUDA

You must manually install a version of PyTorch that matches your local CUDA version (e.g., CUDA 12.4). The requirements file does not handle this.

# Example for CUDA 12.4
```console
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. External Repositories

The pipeline relies on several private or external libraries that must be manually cloned and installed:

* asdf: https://github.com/MillionConcepts/asdf.git
* pdr: https://github.com/MillionConcepts/pdr.git
* silencio: https://github.com/MillionConcepts/silencio.git
* prettyplot: https://github.com/MillionConcepts/pretty-plot.git

### 3. Standard Dependencies

Install the remaining standard Python packages (NumPy, OpenCV, Segment Anything, etc.) via the requirements file:

```console
pip install -r requirements.txt
```

### 4. SAM Model Weights

Ensure you have the Segment Anything Model (SAM) weights.

* Download: ViT-H SAM model (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* Placement: Place in your project root or defined model directory.

## Quick Start

The primary entry point is the Sparc class.

```python
from src.sparc import Sparc, export_spectra_csv

# Initialize
sparc = Sparc(
    sam_model_path="./models/sam_vit_h_4b8939.pth",
    use_gpu=True,
    use_threading=True,
    verbose=True  # Shows shapes, timings, and spectral stats
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
    .analyze(contamination=0.1)
    .select()
)

# Visualize results
sparc.plot(figsize=(15, 12))

# Access results
result = sparc.result
print(f"Found {len(result.final_rois)} ROIs in {result.n_clusters} spectral clusters")

# Export
export_spectra_csv(result, "output/spectra.csv")
```

### Configuration
Configuration is handled via strongly-typed dataclasses in config.py. You can adjust these dynamically via the method arguments shown above, or by modifying the config object directly before running steps.

Tuning params in [`config.py`](src/sparc/core/config.py):
* `albedo_ratio_threshold` (default: 0.80): filters pixels with large descrepancy between the left and right cameras.
* `allowed_variance`: threshold for splitting a SAM segment into multiple spectral clusters.
* `edge_offset`: number of pixels to ignore around the image border.
* `max_subclusters`: Absolute limit on sub-clusters per segment to prevent fragmentation.
