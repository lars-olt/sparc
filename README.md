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

### Install the source tools

Install [Git](https://git-scm.com/downloads), then install
[uv](https://docs.astral.sh/uv/getting-started/installation/) using the command for
your platform. Skip this if `git --version` and `uv --version` already work.
uv will download Python 3.11 when it creates the environment.

**Windows (PowerShell):**

```powershell
winget install --id astral-sh.uv -e
```

**macOS (Terminal):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reopen your terminal after installation and check `uv --version` before continuing.

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

Before running Python commands, activate the environment from `sparc`:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS (Terminal): `source .venv/bin/activate`

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
right images. SPARC handles the matching and maps ROIs into inscribed rectangles in the
other eye. ROIStudio uses this same implementation.

Supported platforms are **64-bit Windows (CPU or NVIDIA CUDA)** and **Apple
Silicon macOS (CPU or MPS)**. On Apple Silicon, use a native ARM64 terminal,
without Rosetta. Intel Macs lack the required PyTorch wheels; see the
[PyTorch support notice](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).

For a new setup, first [install Git and uv](#install-the-source-tools), then follow
the steps below. If RoMa already works, go straight to [launching](#5-launch-with-roma).

### 1. Create and activate the environment

Clone the source and install the pipeline dependencies:

```bash
git clone https://github.com/lars-olt/sparc.git
cd sparc
uv sync --python 3.11 --extra algorithm --inexact --no-install-package torch --no-install-package torchvision
```

For an existing source installation, close any running app, skip cloning, and
run the `uv sync` command from `sparc` to reuse its Python 3.11 `.venv`.
`--inexact` keeps extra packages. PyTorch is installed separately in the next
step because RoMa needs a newer version than the standard environment.

Activate `.venv` from the repository directory:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS (Terminal):**

```bash
source .venv/bin/activate
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and try again.
This applies only to the current terminal.

Keep this terminal open for the remaining steps. Check which Python is active:

```bash
python -c "import sys; print(sys.executable)"
```

The path must point inside this repository’s `.venv`.

### 2. Install PyTorch

Run **one** command for your platform:

**Windows — CPU:**

```bash
uv pip install --python .venv --reinstall "numpy<2" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

**Windows — NVIDIA GPU:** requires a compatible GPU and an up-to-date NVIDIA
driver. The wheel includes the CUDA runtime.

```bash
uv pip install --python .venv --reinstall "numpy<2" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**macOS — Apple Silicon:** supports both CPU and MPS.

```bash
uv pip install --python .venv --reinstall "numpy<2" torch==2.6.0 torchvision==0.21.0
```

These use the [PyTorch 2.6 builds](https://pytorch.org/get-started/previous-versions/#v260).
`--reinstall` also handles switching an existing environment between CPU and CUDA.

### 3. Install requirements-roma.txt (required)

From `sparc`, install RoMa into the same `.venv`:

```bash
uv pip install --python .venv -r requirements-roma.txt
```

Wait for installation to finish, then verify the import and selected device:

```bash
python -c "from romatch import roma_outdoor; from sparc.utils.device import resolve_device; print('RoMa device:', resolve_device('auto'))"
```

### 4. Prepare the model weights

**SAM:** download `sam_vit_h_4b8939.pth` from
[Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and keep it in a permanent location. This checkpoint is required for automatic
ROI generation. Reuse your existing file if you already have it.

**RoMa:** the `roma_outdoor.pth` and `dinov2_vitl14_pretrain.pth` weights are
required, but download automatically when RoMa first loads a scene. Allow internet
access and several gigabytes of disk space for this first run. Installing
`requirements-roma.txt` installs the software; the weight download happens later.

PyTorch caches the weights outside `.venv`, normally in
`~/.cache/torch/hub/checkpoints` (`~` is your user folder). Existing weights in
that cache are reused across environments for the same user. To see the actual
cache folder, including any `TORCH_HOME` override, run:

```bash
python -c "from pathlib import Path; import torch; print(Path(torch.hub.get_dir()) / 'checkpoints')"
```

To download and check RoMa’s weights before launching, or before going offline,
run this once while connected. It uses CPU and reuses cached files:

```bash
python -c "from romatch import roma_outdoor; roma_outdoor(device='cpu', use_custom_corr=False); print('RoMa weights ready')"
```

Wait for `RoMa weights ready`. The local-correlation warning on Windows and macOS
is expected and does not prevent setup.

### 5. Launch with RoMa

From `sparc`, with `.venv` active, replace the two input paths with your image
folder and SAM checkpoint:

```bash
python -m sparc --input "path/to/iof" --sam-path "path/to/sam_vit_h_4b8939.pth" --alignment roma --device auto --obs-index 0 --output "sparc-results/roma-scene"
```

`auto` chooses CUDA, then MPS, then CPU. Use `--device cpu`, `--device cuda`, or
`--device mps` to choose explicitly. An unavailable requested GPU raises an error;
unsupported MPS operations can fall back to CPU.

The output folder must be new. It will contain the overview image, spectra, ROIs,
and settings. See [Terminal use](#terminal-use) for scene selection and YAML
configuration. For the GUI, follow the
[ROIStudio setup](https://github.com/lars-olt/roistudio#experimental-roma).

**On later launches:** open a terminal in the repository, activate `.venv` using
the command in step 1, and run the launch command above. Dependencies and cached
weights do not need reinstalling. Run `deactivate` when finished.

If you prefer to skip activation, use `uv run --no-sync python` in place of
`python`. Always include `--no-sync`: plain `uv run` or `uv sync` can restore the
standard pins and remove RoMa or downgrade PyTorch. If that happens, repeat
steps 2 and 3.

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

With `.venv` active, run a scene from the terminal with `python -m sparc`. The `sparc` command is also
available after installing the package. Use `--help` to list the options.

```powershell
python -m sparc --input "path/to/iof" --sam-path "path/to/sam_vit_h_4b8939.pth" --alignment roma --device auto --obs-index 0 --output "sparc-results/roma-scene"
```

`--obs-index` is the zero-based pointing index, using the same grouping as
ROIStudio. Add `--seq-id` to select a sequence before indexing its pointings.
`--device cuda` requires a CUDA-enabled PyTorch build and a compatible NVIDIA GPU for SAM and RoMa. Use
`--device cpu` for CPU execution, `--device mps` for an Apple Silicon GPU,
or `--device auto` to choose CUDA, then MPS, then CPU.
Homography is the default alignment.

For more settings, use [examples/roma.yml](examples/roma.yml):

```powershell
python -m sparc --config examples/roma.yml --input "path/to/iof" --sam-path "path/to/sam_vit_h_4b8939.pth"
python -m sparc --config examples/roma.yml --roma-certainty 0.6 --print-config
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
