# SPARC: Spectral Pattern Analysis for ROI Classification

[![CI/CD](https://github.com/lars-olt/sparc/actions/workflows/ci.yml/badge.svg)](https://github.com/lars-olt/sparc/actions/workflows/ci.yml)

SPARC extracts regions of interest from Mastcam-Z and Pancam multispectral images. It uses SAM to segment the images, finds rectangular ROIs within the segments, and groups them by their spectra.

## Architecture

The pipeline code is in `src/sparc/core`:

* [sparc.py](src/sparc/core/sparc.py): The `Sparc` class used to run the pipeline step by step.
* [pipeline.py](src/sparc/core/pipeline.py): Loading, preprocessing, segmentation, ROI extraction, and spectral analysis.
* [state.py](src/sparc/core/state.py): Data passed between steps.
* [backends.py](src/sparc/core/backends.py): CPU/GPU segmentation and sequential/threaded ROI processing.

## Installation

For RoMa, follow [Experimental RoMa](#experimental-roma) instead of the standard setup below.

### 1. Clone and install

```bash
git clone https://github.com/lars-olt/sparc.git
cd sparc
uv venv
uv sync
```

The base install includes loading, ROI handling, spectra, plotting, and file I/O.
To run segmentation and spectral clustering, install the algorithm dependencies:

```bash
uv sync --extra algorithm
```

For projects that use SPARC as a dependency:

```bash
pip install "sparc @ git+ssh://git@github.com/lars-olt/sparc.git"
pip install "sparc[algorithm] @ git+ssh://git@github.com/lars-olt/sparc.git"
```

Use `sparc[algorithm]` for the full pipeline, or `sparc` for the base package.

### 2. GPU acceleration (optional)

SPARC uses the CPU by default. For GPU segmentation, install a CUDA build of
PyTorch and set `use_gpu=True`. Use the command for your system from
[PyTorch](https://pytorch.org/get-started/locally/). For example, with CUDA 12.1:

```bash
# example for CUDA 12.1 - replace cu121 with your version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 3. SAM model checkpoint

Download `sam_vit_h_4b8939.pth` from the [Segment Anything repository](https://github.com/facebookresearch/segment-anything). Pass its path as `sam_model_path` when creating a `Sparc` instance.

## Experimental RoMa

RoMa is an experimental alternative to homography for aligning the left and
right images. SPARC handles the matching and maps ROIs into inscribed
rectangles in the other eye. ROIStudio uses this same implementation.

This setup supports **Windows (64-bit), with CPU or NVIDIA CUDA**, and
**Apple Silicon macOS, with CPU or MPS**. Intel Macs cannot use this setup:
[official PyTorch wheels for Intel Macs ended after 2.2](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690),
and RoMa needs a newer version.

### 1. Set up the environment

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then open PowerShell
on Windows or Terminal on macOS. Use these instructions instead of the standard
installation above. uv creates a Python 3.11 environment in `.venv` and downloads
Python if needed. On Apple Silicon, use a native ARM64 terminal, without Rosetta.

```bash
git clone https://github.com/lars-olt/sparc.git
cd sparc
uv sync --python 3.11 --extra algorithm --inexact --no-install-package torch --no-install-package torchvision
```

For an existing source installation, skip cloning, open the `sparc` directory,
and run the `uv sync` command above to reuse its Python 3.11 `.venv`. Deactivate
any other environment before following these steps. `--inexact` keeps additional
packages you have installed.

The command installs the pipeline dependencies but leaves PyTorch for the next
step. The standard environment pins an older PyTorch version; RoMa uses 2.6.

### 2. Install PyTorch

Run **one** of these commands from the same directory:

**Windows — CPU:**

```bash
uv pip install --python .venv --reinstall "numpy<2" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

**Windows — NVIDIA GPU:** use this with a CUDA-compatible GPU and an up-to-date
NVIDIA driver. The wheel includes the CUDA runtime.

```bash
uv pip install --python .venv --reinstall "numpy<2" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**macOS — Apple Silicon:** the same build supports CPU and MPS.

```bash
uv pip install --python .venv --reinstall "numpy<2" torch==2.6.0 torchvision==0.21.0
```

These are the [PyTorch 2.6 installation builds](https://pytorch.org/get-started/previous-versions/#v260).
`--reinstall` also handles switching an existing installation between CPU and CUDA.

### 3. Install requirements-roma.txt (required)

From the `sparc` directory, run this command on Windows or macOS to install
RoMa and its dependencies into `sparc/.venv`:

```bash
uv pip install --python .venv -r requirements-roma.txt
```

Complete this step after installing PyTorch, including when updating an existing
environment. Wait for the installation to succeed before continuing.

### 4. Check and run

Check the Python path, RoMa import, and selected device. The printed Python
path should be inside this checkout’s `sparc/.venv`:

```bash
uv run --no-sync python -c "import sys; print('Python:', sys.executable); from romatch import roma_outdoor; from sparc.utils.device import resolve_device; print('RoMa device:', resolve_device('auto'))"
```

`auto` chooses CUDA, then MPS, then CPU. Use `--device cpu` to run without a GPU,
`--device cuda` for an NVIDIA GPU, or `--device mps` for an Apple Silicon GPU.
If the check reports `cpu` when you expected a GPU, check the PyTorch build and
driver or macOS support before running. An explicitly requested unavailable GPU
raises an error. Unsupported MPS operations can fall back to CPU.

Download `sam_vit_h_4b8939.pth` from
[Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints)
if you do not already have it. Run the full pipeline, replacing the two input
paths with your image folder and checkpoint:

```bash
uv run --no-sync python -m sparc --input "path/to/iof" --sam-path "path/to/sam_vit_h_4b8939.pth" --alignment roma --device auto --obs-index 0 --output "sparc-results/roma-scene"
```

RoMa downloads its own weights on first use, so the first run needs internet
access and takes longer. The output folder must be new; it will contain the
overview image, spectra, ROIs, and run settings. See [Terminal use](#terminal-use)
for scene selection and YAML configuration.

For the desktop interface, follow the
[ROIStudio RoMa setup](https://github.com/lars-olt/roistudio#experimental-roma).
Its environment includes SPARC and can run both the GUI and terminal pipeline.

**Keep `--no-sync` when launching.** A plain `uv run` or `uv sync` restores the
standard dependency pins and can remove RoMa or downgrade PyTorch. If that
happens, repeat steps 2 and 3. You can also activate `.venv` and run `python` directly.

## Quick Start

Use the `Sparc` class to run each step and inspect the results:

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

Use `run_sparc` to run all steps in one call:

```python
from sparc.core.functional import run_sparc
from sparc.core.config import SparcConfig, LoadConfig, SegmentConfig

config = SparcConfig(
    load    = LoadConfig(iof_path="/path/to/data", instrument="ZCAM"),
    segment = SegmentConfig(sam_model_path="./models/sam_vit_h_4b8939.pth"),
)
result = run_sparc(
    iof_path=config.load.iof_path,
    sam_model_path=config.segment.sam_model_path,
    config=config,
)
```

### Terminal use

Run a scene from the terminal with `python -m sparc`. The `sparc` command is also
available after installing the package. Use `--help` to list the options.

```powershell
uv run --no-sync python -m sparc --input "path/to/iof" --sam-path "path/to/sam_vit_h_4b8939.pth" --alignment roma --device auto --obs-index 0 --output "sparc-results/roma-scene"
```

`--obs-index` is the zero-based pointing index, using the same grouping as
ROIStudio. Add `--seq-id` to select a sequence before indexing its pointings.
`--device cuda` requires a CUDA-enabled PyTorch build and a compatible NVIDIA GPU for SAM and RoMa. Use
`--device cpu` for CPU execution, `--device mps` for an Apple Silicon GPU,
or `--device auto` to choose CUDA, then MPS, then CPU.
Homography is the default alignment.

For more settings, use [examples/roma.yml](examples/roma.yml):

```powershell
uv run --no-sync python -m sparc --config examples/roma.yml --input "path/to/iof" --sam-path "path/to/sam_vit_h_4b8939.pth"
uv run --no-sync python -m sparc --config examples/roma.yml --roma-certainty 0.6 --print-config
```

Command-line options override the YAML file. `--print-config` shows the settings
without loading images or models. The example uses ROIStudio's default algorithm
settings, including background preservation and threaded ROI extraction.

Each run saves the settings, left/right rectangles, segment labels, spectra,
and an overview plot. Spectra are SPARC's calibrated pipeline values; ROIStudio
also recalculates display spectra from the raw per-camera rectangles. Output
directories must be new so an earlier run is not overwritten.

### Pipeline settings

Settings are defined in [`config.py`](src/sparc/core/config.py). Set them through the method arguments above or edit the config before running a step.

Parameters to adjust:

- `albedo_ratio_threshold` (default: `0.80`) - filters ROIs with a large brightness discrepancy between left and right cameras. ZCAM only.
- `allowed_variance` (default: `1.0`) - threshold for splitting a SAM segment into multiple spectral subclusters. Lower values produce finer splits.
- `edge_offset` (default: `10`) - pixels ignored around the image border to avoid edge artifacts.
- `max_subclusters` (default: `10`) - maximum number of subclusters per segment.
- `max_components` (default: `9`) - maximum number of spectral clusters the Bayesian GMM may find.

## Development

Run the tests without a SAM checkpoint:

```bash
uv pip install "pytest>=8,<10"
uv run --no-sync pytest -q
```

CI runs the tests on Windows and Apple Silicon and checks the wheel and source
packages. To make a release, use a version tag such as `v2.0.0` that matches
`pyproject.toml`. The release is published after the tests and package checks pass.
