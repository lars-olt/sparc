# SPARC: Spectral Pattern Analysis for ROI Classification

SPARC is a modular pipeline for automating Region of Interest (ROI) extraction from hyperspectral imagery (specifically ZCAM/Marslab data). It combines deep learning (SAM) for semantic segmentation with statistical spectral clustering to identify homogeneous geological features.

## Architecture

The project follows a **Functional Core, Imperative Shell** pattern:

* **Shell (sparc.py):** The Object-Oriented interface for state management and configuration.
* **Core (pipeline.py):** Pure functions transforming SparcState through distinct processing steps.
* **State (state.py):** A dataclass acting as the single source of truth for the pipeline lifecycle.
* **Backends (backends.py):** Abstraction layer for swapping segmentation (CPU/GPU) and clustering implementations.

## Installation & Setup

This project requires manual configuration of specific dependencies and external repositories before running.

### 1. PyTorch & CUDA

You must manually install a version of PyTorch that matches your local CUDA version (e.g., CUDA 12.4). The standard requirements file does not handle this to avoid platform mismatches.

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

# 1. Initialize
# 'verbose=True' enables debug logging (shapes, timings, spectral stats)
sparc = Sparc(
    sam_model_path="./models/sam_vit_h_4b8939.pth",
    use_gpu=True,       # Auto-falls back to CPU if CUDA unavailable
    use_threading=True,
    verbose=True
)

# 2. Run Pipeline
# Supports method chaining
(sparc
    .load(iof_path="/path/to/mcz/data", obs_ix=0)
    .preprocess(apply_r_star=True)
    .segment(points_per_side=32)
    .extract_rois(
        area_threshold=50,
        min_cluster_area=500,  # Min area to attempt sub-clustering
        min_clean_area=4000    # Min area to keep after morphological cleaning
    )
    .analyze(contamination=0.1) # Outlier detection
    .select()
)

# 3. Visualization & Export
# Plot summary (RGB, Segmentation, ROIs, Spectra)
sparc.plot(figsize=(15, 12))

# Access immutable results
result = sparc.result
print(f"Found {len(result.final_rois)} ROIs in {result.n_clusters} spectral clusters")

# Export to Marslab-compatible CSV
export_spectra_csv(result, "output/spectra.csv")
```

### Configuration
Configuration is handled via strongly-typed dataclasses in config.py. You can adjust these dynamically via the method arguments shown above, or by modifying the config object directly before running steps.

Key Tuning Parameters:
* ROIConfig:
    * albedo_ratio_threshold: Filters artifacts where Left/Right camera alignment fails (default: 0.80).
    * allowed_variance: Threshold for splitting a SAM segment into multiple spectral clusters.
    * edge_offset: Pixels to ignore around the image border.
    * max_subclusters: Absolute limit on sub-clusters per segment to prevent fragmentation.

### Technical Notes
* Threading on Windows: This project utilizes a SafeKMeans wrapper in threading.py. It explicitly manages environment variables (setting OMP_NUM_THREADS=1 for worker processes) to prevent memory leaks and crashes associated with the MKL library when using Python's multiprocessing on Windows.
* Dependencies: Requires marslab (for IOF data loading), segment-anything, torch, scikit-learn, numpy, and matplotlib.