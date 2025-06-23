# SPARC: Spectral Pattern Analysis for ROI Classification

SPARC is a Python package for analyzing hyperspectral images to identify and classify regions of interest (ROIs) based on their spectral signatures.

## Installation

### Requirements

- **Python 3.11** (required)
- Git (for installing dependencies from repositories)

### Option 1: uv (Recommended - Fast)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/lars-olt/sparc.git
cd sparc

# Create environment and install (uv handles everything automatically)
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or install in existing environment
uv pip install -e .
```

### Option 2: pip (Standard)

```bash
# Clone the repository
git clone https://github.com/lars-olt/sparc.git
cd sparc

# Create virtual environment with Python 3.11
python3.11 -m venv sparc-env
source sparc-env/bin/activate  # On Windows: sparc-env\Scripts\activate

# Install SPARC with all dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 3: Direct pip install

```bash
# Create virtual environment
python3.11 -m venv sparc-env
source sparc-env/bin/activate

# Install directly from repository
pip install git+https://github.com/lars-olt/sparc.git

# Or with development tools
pip install "git+https://github.com/lars-olt/sparc.git[dev]"
```

### SAM Model Download

Download the required SAM model weights:

```bash
# Create models directory
mkdir models
cd models

# Download ViT-H model (recommended, ~2.4GB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Quick Start

```python
from sparc import Sparc

# Initialize with SAM model path
sparc = Sparc(sam_model_path="models/sam_vit_h_4b8939.pth")

# Run complete pipeline
sparc.run_pipeline(
    iof_path="path/to/data",
    obs_ix=0,
    load_kwargs={'do_apply_pixmaps': True},
    preprocess_kwargs={
        'shadow_kwargs': {'percentiles': (20, 100)},
        'apply_r_star': True
    }
)

# Access results
print(sparc.summary)
sparc.plot_results()
```

## Troubleshooting

### KMeans Memory Leak Warning (Windows)

If you see warnings about KMeans memory leaks on Windows:

```python
from sparc import force_fix_kmeans_warnings
force_fix_kmeans_warnings()  # Call before using SPARC
```

## Step-by-Step Usage

For more control over the pipeline:

```python
# Load data
sparc.load_data("path/to/data", obs_ix=0)

# Preprocess
sparc.preprocess()

# Segment image
sparc.segment()

# Extract ROIs
sparc.extract_rois()

# Filter and analyze
sparc.filter_rois(area_threshold=50)
sparc.analyze_spectra()
sparc.select_final_rois()

# Visualize
sparc.plot_results()
```

## Export and Analysis Results

SPARC provides export capabilities for saving results:

### Quick Export (All Formats)

```python
# Export everything with one command
exported_files = sparc.export_results(
    output_dir="results",
    base_name="my_analysis",
    metadata={'SOL': 619, 'FEATURE': 'bedrock'}
)
```

### Individual Exports

#### CSV Export (marslab-compatible format)

```python
# Save ROI data in marslab CSV format
metadata = {
    'SOL': 619,
    'SITE': 42,
    'FEATURE': 'bedrock',
    'DESCRIPTION': 'Automated ROI detection',
    'LAT': 18.4446,
    'LON': 77.4509
}
sparc.save_roi_csv("roi_analysis.csv", metadata)
```

#### Plot and Image Export

```python
# Save high-resolution plots
plots = sparc.save_plots("output_dir/", dpi=300)

# Or use SparcExporter for more control
from sparc.utils.io import SparcExporter
exporter = SparcExporter(sparc)

# Save spectra plot as PDF for publication
exporter.save_spectra_plot("spectra.pdf", figsize=(10,6), dpi=300, format='pdf')

# Save ROI context image
exporter.save_roi_context_image("context.png", dpi=600)

# Save complete pipeline summary
exporter.save_pipeline_summary_image("pipeline.png")
```

## Exported File Formats

1. **CSV File**: marslab-compatible format with spectral data, ROI coordinates, and metadata
2. **Spectra Plot**: High-resolution plot of final selected spectra
3. **ROI Context Image**: RGB image with ROI overlays
4. **Pipeline Summary**: 4-panel summary showing segmentation, ROIs, and spectra
5. **Results Pickle**: Complete pipeline state for later analysis

## Pipeline Components

### Data Loading

- ZCAM hyperspectral data ingestion
- Homography correction for stereo alignment
- RGB image generation for segmentation

### Preprocessing

- Shadow and sky masking
- Bad pixel correction via pixmaps
- Photometric calibration (R\* conversion)

### Segmentation

- SAM (Segment Anything Model) integration
- Automatic region detection

### ROI Analysis

- Spectral clustering within segments
- Area-based filtering
- Outlier detection using frequency analysis
- Bayesian Gaussian Mixture Models

### Visualization

- ROI overlay plots
- Spectral signature plots
- Clustering results visualization
- Complete pipeline summary plots

## API Reference

### Main Class

#### `Sparc(sam_model_path)`

Main class for the SPARC pipeline.

**Methods:**

- `load_data()`: Load hyperspectral data
- `preprocess()`: Apply masking and calibration
- `segment()`: Perform SAM segmentation
- `extract_rois()`: Extract potential ROIs
- `filter_rois()`: Filter ROIs by area and spectral properties
- `analyze_spectra()`: Perform spectral clustering analysis
- `select_final_rois()`: Apply final selection heuristics
- `plot_results()`: Visualize final results
- `run_pipeline()`: Execute complete pipeline

**State Management:**

- `save_state()`: Save current pipeline state to file
- `reload_from_file()`: Load state from saved file
- `continue_analysis()`: Continue analysis from saved state
- `compare_with_saved()`: Compare current results with saved file

**Class Methods:**

- `from_saved_results()`: Create instance from saved file

**Export Methods:**

- `export_results()`: Export all results (CSV, plots, images)
- `save_roi_csv()`: Save ROI data in marslab format
- `save_plots()`: Save visualization plots

**Properties:**

- `summary`: Pipeline state summary
- `is_complete`: Whether pipeline is finished
- `is_loaded_from_file`: Whether instance was loaded from file
- `final_rois`: Final selected ROIs
- `roi_spectra`: Extracted spectra
- `clustering_result`: Clustering analysis results

## Development

For contributors working on SPARC:

```bash
# Clone repository
git clone https://github.com/lars-olt/sparc.git
cd sparc

# Install with development dependencies
uv sync  # if using uv
# or
pip install -e ".[dev]"  # if using pip

# Run tests
pytest tests/

# Code formatting
black src/
isort src/
```
